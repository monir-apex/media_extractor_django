"""Tests for face matching. The face models are replaced by fakes, so these run without them."""
from datetime import timedelta
from unittest import mock

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from . import faces, matching
from .models import DownloadBatch, MatchJob, Photo, PhotoFace, Profile

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


def unit(*values):
    vector = np.zeros(faces.EMBEDDING_SIZE, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def face(embedding, x=0.1, size=0.2):
    return faces.Face(x=x, y=0.1, width=size, height=size, confidence=0.95, embedding=embedding)


ME = unit(1, 0)
SOMEONE_ELSE = unit(0, 1)
ALMOST_ME = unit(1, 0.3)  # cosine ~0.96 with ME


@override_settings(FACE_MATCH_PERCENT=70)
class PercentTests(SimpleTestCase):
    def test_threshold_maps_to_match_percent(self):
        self.assertEqual(faces.similarity_to_percent(faces.SAME_PERSON_SIMILARITY), 70)
        self.assertEqual(faces.similarity_to_percent(faces.SAME_PERSON_SIMILARITY - 0.001), 69)

    def test_scale_is_clamped(self):
        self.assertEqual(faces.similarity_to_percent(-0.4), 0)
        self.assertEqual(faces.similarity_to_percent(0.0), 0)
        self.assertEqual(faces.similarity_to_percent(faces.FULL_MATCH_SIMILARITY), 100)
        self.assertEqual(faces.similarity_to_percent(1.0), 100)

    def test_scale_increases(self):
        values = [faces.similarity_to_percent(s / 100) for s in range(0, 101)]
        self.assertEqual(values, sorted(values))

    def test_embedding_round_trip(self):
        np.testing.assert_array_equal(faces.embedding_from_bytes(faces.embedding_to_bytes(ME)), ME)


class ProfileFaceTests(SimpleTestCase):
    def test_single_face(self):
        with mock.patch.object(faces, "find_faces", return_value=[face(ME)]):
            np.testing.assert_array_equal(faces.profile_face(b"img").embedding, ME)

    def test_no_face(self):
        with mock.patch.object(faces, "find_faces", return_value=[]):
            with self.assertRaisesMessage(faces.ProfileFaceError, "No face was found"):
                faces.profile_face(b"img")

    def test_clearly_largest_face_wins(self):
        found = [face(ME, size=0.4), face(SOMEONE_ELSE, size=0.1)]
        with mock.patch.object(faces, "find_faces", return_value=found):
            np.testing.assert_array_equal(faces.profile_face(b"img").embedding, ME)

    def test_two_similar_faces_are_rejected(self):
        found = [face(ME, size=0.3), face(SOMEONE_ELSE, size=0.28)]
        with mock.patch.object(faces, "find_faces", return_value=found):
            with self.assertRaisesMessage(faces.ProfileFaceError, "2 faces"):
                faces.profile_face(b"img")


class MatchTestCase(TestCase):
    def setUp(self):
        self.batch = DownloadBatch.objects.create(source_url="https://example.com/a", provider="dropbox")
        Profile.objects.create(pk=1, display_name="Monir", photo=b"me", photo_content_type="image/jpeg")
        # What the fake detector "sees" in each photo, keyed by the photo's bytes.
        self.scene = {
            b"me": [face(ME)],
            b"group": [face(SOMEONE_ELSE, x=0.1), face(ALMOST_ME, x=0.5), face(SOMEONE_ELSE, x=0.8)],
            b"strangers": [face(SOMEONE_ELSE)],
            b"landscape": [],
        }
        patcher = mock.patch.object(faces, "find_faces", side_effect=self.fake_find_faces)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(faces, "models_available", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_find_faces(self, data):
        data = bytes(data)
        if data not in self.scene:
            raise faces.FaceError("Could not read the image")
        return self.scene[data]

    def add_photo(self, name, data):
        return Photo.objects.create(
            batch=self.batch, filename=name, content_type="image/jpeg", size=len(data), sha256=name, data=data
        )

    def sync(self):
        job = MatchJob.objects.create()
        matching.run_sync(job.pk)
        job.refresh_from_db()
        return job


class SyncTests(MatchTestCase):
    def test_group_photo_scores_its_best_face(self):
        group = self.add_photo("group.jpg", b"group")
        strangers = self.add_photo("strangers.jpg", b"strangers")
        landscape = self.add_photo("landscape.jpg", b"landscape")

        job = self.sync()

        self.assertEqual(job.status, MatchJob.DONE, job.error)
        self.assertEqual((job.total_photos, job.photos_with_faces, job.matched), (3, 2, 1))
        self.assertEqual(job.percent, 100)
        group.refresh_from_db()
        self.assertTrue(group.is_match)
        self.assertGreaterEqual(group.match_percent, 95)
        self.assertEqual(group.matched_face.x, 0.5)  # the middle face is the profile person
        self.assertEqual(group.faces.count(), 3)
        strangers.refresh_from_db()
        self.assertFalse(strangers.is_match)
        self.assertEqual(strangers.match_percent, 0)
        landscape.refresh_from_db()
        self.assertTrue(landscape.faces_scanned)
        self.assertIsNone(landscape.match_percent)

    def test_second_sync_only_scans_new_photos(self):
        self.add_photo("group.jpg", b"group")
        self.sync()
        self.add_photo("me.jpg", b"me")
        job = self.sync()
        self.assertEqual((job.to_scan, job.scanned, job.matched), (1, 1, 2))
        self.assertEqual(faces.find_faces.call_count, 3)  # profile, group, me

    def test_unreadable_photos_are_skipped(self):
        self.add_photo("broken.jpg", b"broken")
        self.add_photo("group.jpg", b"group")
        job = self.sync()
        self.assertEqual(job.status, MatchJob.DONE)
        self.assertEqual((job.unreadable, job.matched), (1, 1))

    def test_profile_without_face_fails(self):
        Profile.objects.filter(pk=1).update(photo=b"landscape")
        self.add_photo("group.jpg", b"group")
        job = self.sync()
        self.assertEqual(job.status, MatchJob.FAILED)
        self.assertIn("No face was found", job.error)

    def test_missing_models_are_downloaded_first(self):
        self.add_photo("group.jpg", b"group")
        with mock.patch.object(faces, "models_available", return_value=False), \
                mock.patch.object(faces, "ensure_models") as ensure:
            job = self.sync()
        ensure.assert_called_once()
        self.assertEqual(job.status, MatchJob.DONE)

    def test_model_download_failure_is_reported(self):
        with mock.patch.object(faces, "models_available", return_value=False), \
                mock.patch.object(faces, "ensure_models", side_effect=faces.FaceError("offline")):
            job = self.sync()
        self.assertEqual((job.status, job.error), (MatchJob.FAILED, "offline"))

    def test_deleting_a_photo_deletes_its_faces(self):
        group = self.add_photo("group.jpg", b"group")
        self.sync()
        group.delete()
        self.assertFalse(PhotoFace.objects.exists())

    def test_stale_job_is_marked_failed(self):
        job = MatchJob.objects.create(status=MatchJob.RUNNING)
        MatchJob.objects.filter(pk=job.pk).update(updated_at=timezone.now() - timedelta(hours=2))
        job = matching.refresh_if_stale(MatchJob.objects.get(pk=job.pk))
        self.assertEqual(job.status, MatchJob.FAILED)


class SyncViewTests(MatchTestCase):
    def test_sync_button_starts_job(self):
        self.add_photo("group.jpg", b"group")
        with mock.patch.object(matching, "_run_in_thread") as run:
            response = self.client.post("/photos/match/sync/", HTTP_ACCEPT="application/json")
        self.assertEqual(response.status_code, 202)
        job = MatchJob.objects.get()
        self.assertEqual(response.json()["id"], str(job.pk))
        self.assertEqual(run.call_count, 1)

        # Pressing it again while running returns the same job.
        with mock.patch.object(matching, "_run_in_thread") as run:
            again = self.client.post("/photos/match/sync/", HTTP_ACCEPT="application/json")
        self.assertEqual(again.json()["id"], str(job.pk))
        run.assert_not_called()

        status = self.client.get(f"/api/photos/match/jobs/{job.pk}/")
        self.assertEqual(status.json()["status"], MatchJob.QUEUED)

    def test_sync_needs_profile_photo(self):
        Profile.objects.filter(pk=1).update(photo=None, photo_content_type="")
        response = self.client.post("/photos/match/sync/", HTTP_ACCEPT="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Add a profile photo first", response.json()["error"])

    def test_sync_rejects_get(self):
        self.assertEqual(self.client.get("/photos/match/sync/").status_code, 405)

    def test_gallery_shows_scores_and_matches_view(self):
        group = self.add_photo("group.jpg", b"group")
        self.add_photo("strangers.jpg", b"strangers")
        self.add_photo("landscape.jpg", b"landscape")
        self.sync()
        group.refresh_from_db()

        gallery = self.client.get("/photos/")
        self.assertContains(gallery, "Sync face match")
        self.assertContains(gallery, f"{group.match_percent}%")
        self.assertContains(gallery, "match-badge is-match", count=1)
        self.assertContains(gallery, "Matches (1)")
        self.assertContains(gallery, f'data-face="{group.matched_face.box_percent}"')
        self.assertContains(gallery, "<strong>1 photo</strong> matches")

        matches = self.client.get("/photos/?view=matches")
        self.assertContains(matches, "group.jpg")
        self.assertNotContains(matches, "strangers.jpg")
        self.assertNotContains(matches, "landscape.jpg")

    def test_gallery_shows_running_job(self):
        self.add_photo("group.jpg", b"group")
        job = MatchJob.objects.create(status=MatchJob.RUNNING, stage="Finding faces… photo 1 of 1", percent=40)
        gallery = self.client.get("/photos/")
        self.assertContains(gallery, "Finding faces… photo 1 of 1")
        self.assertContains(gallery, f"/api/photos/match/jobs/{job.pk}/")
        self.assertContains(gallery, "width: 40%")


class ProfilePhotoChangeTests(MatchTestCase):
    def upload(self, data):
        return SimpleUploadedFile("me.png", PNG + data, content_type="image/png")

    def setUp(self):
        super().setUp()
        self.scene[PNG + b"me"] = [face(ME)]
        self.scene[PNG + b"nobody"] = []
        self.scene[PNG + b"two"] = [face(ME, size=0.3), face(SOMEONE_ELSE, size=0.3)]

    def test_new_photo_computes_face_and_clears_old_scores(self):
        group = self.add_photo("group.jpg", b"group")
        self.sync()
        response = self.client.post("/profile/", {"display_name": "Monir", "photo": self.upload(b"me")})
        self.assertRedirects(response, "/profile/")
        np.testing.assert_array_equal(faces.embedding_from_bytes(Profile.load().face_embedding), ME)
        group.refresh_from_db()
        self.assertIsNone(group.match_percent)
        self.assertTrue(group.faces_scanned)  # faces are kept, only scores are cleared

    def test_photo_without_face_is_rejected(self):
        response = self.client.post("/profile/", {"display_name": "Monir", "photo": self.upload(b"nobody")})
        self.assertContains(response, "No face was found")
        self.assertEqual(bytes(Profile.load().photo), b"me")  # old photo kept

    def test_photo_with_several_faces_is_rejected(self):
        response = self.client.post("/profile/", {"display_name": "Monir", "photo": self.upload(b"two")})
        self.assertContains(response, "2 faces")

    def test_removing_photo_clears_scores(self):
        group = self.add_photo("group.jpg", b"group")
        self.sync()
        self.client.post("/profile/", {"action": "remove-photo"})
        group.refresh_from_db()
        self.assertIsNone(group.match_percent)
        self.assertIsNone(Profile.load().face_embedding)
