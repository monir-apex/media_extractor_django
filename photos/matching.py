"""Face-match sync: find faces in every photo and score them against the profile face.

A sync runs in a background thread (like downloads in jobs.py) and records its progress in
MatchJob, which the gallery polls. It has two steps:

1. Scan: find and embed the faces in photos that have not been scanned yet. This is the slow
   part, and its results are kept, so later syncs only scan new photos.
2. Score: compare every stored face with the profile face. Each photo's score is its best
   face, so in a group photo the person who looks most like the profile decides the score.
"""
import logging
import threading
import time

import numpy as np
from django.db import close_old_connections, connection, transaction
from django.utils import timezone

from . import faces
from .models import MatchJob, Photo, PhotoFace, Profile

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 0.5
# A running sync with no progress update for this long is assumed dead (server restarted).
# Generous, because the first sync may spend a while downloading the face models.
STALE_AFTER_SECONDS = 30 * 60
# The scan step covers this share of the progress bar; scoring is quick and takes the rest.
SCAN_SHARE = 95


class MatchError(Exception):
    pass


# --- Profile face ---------------------------------------------------------------------

def update_profile_face(profile: Profile, check_now=True):
    """Call after the profile photo changes: forget old scores and the old profile face.

    With check_now, the new photo is checked for a face straight away when the face models
    are already downloaded, so the profile form can explain a bad photo (ProfileFaceError).
    Otherwise the face is found during the next sync.
    """
    profile.face_embedding = None
    if check_now and profile.has_photo and faces.models_available():
        try:
            face = faces.profile_face(profile.photo)
        except faces.ProfileFaceError:
            raise
        except faces.FaceError as exc:
            logger.warning("Could not check the profile photo for a face: %s", exc)
        else:
            profile.face_embedding = faces.embedding_to_bytes(face.embedding)
    clear_scores()


def clear_scores():
    Photo.objects.exclude(match_percent=None, matched_face=None).update(match_percent=None, matched_face=None)


def _profile_embedding(profile: Profile) -> np.ndarray:
    if not profile.has_photo:
        raise MatchError("Add a profile photo first, so there is a face to look for.")
    if not profile.face_embedding:
        try:
            face = faces.profile_face(profile.photo)
        except faces.FaceError as exc:
            raise MatchError(f"Profile photo: {exc}") from exc
        profile.face_embedding = faces.embedding_to_bytes(face.embedding)
        Profile.objects.filter(pk=profile.pk).update(face_embedding=profile.face_embedding)
    return faces.embedding_from_bytes(profile.face_embedding)


# --- Jobs -----------------------------------------------------------------------------

def start_sync() -> MatchJob:
    """Start a sync in the background, or return the one already running."""
    if not Profile.load().has_photo:
        raise MatchError("Add a profile photo first, so there is a face to look for.")
    running = MatchJob.objects.filter(status__in=[MatchJob.QUEUED, MatchJob.RUNNING]).first()
    if running and not refresh_if_stale(running).finished:
        return running
    job = MatchJob.objects.create(stage="Starting…")
    threading.Thread(target=_run_in_thread, args=(job.pk,), daemon=True, name=f"face-match-{job.pk}").start()
    return job


def _run_in_thread(job_id):
    close_old_connections()
    try:
        run_sync(job_id)
    finally:
        connection.close()


def run_sync(job_id):
    update = _job_updater(job_id)
    update(force=True, status=MatchJob.RUNNING)
    try:
        if not faces.models_available():
            update(force=True, stage="Downloading face models (first sync only)…")
            faces.ensure_models()
        update(force=True, stage="Reading the profile face…")
        profile_embedding = _profile_embedding(Profile.load())
        _scan_new_photos(update)
        update(force=True, stage="Comparing faces…", percent=SCAN_SHARE)
        totals = score_photos(profile_embedding)
    except (MatchError, faces.FaceError) as exc:
        update(force=True, status=MatchJob.FAILED, error=str(exc))
    except Exception as exc:  # never leave a job stuck at "running"
        logger.exception("Face match %s failed", job_id)
        update(force=True, status=MatchJob.FAILED, error=f"Unexpected error: {exc}")
    else:
        update(force=True, status=MatchJob.DONE, stage="Done", percent=100, **totals)


def _scan_new_photos(update):
    photo_ids = list(Photo.objects.filter(faces_scanned=False).values_list("id", flat=True))
    total = len(photo_ids)
    update(force=True, stage="Finding faces…", to_scan=total, scanned=0, percent=0 if total else SCAN_SHARE)
    unreadable = 0
    for done, photo_id in enumerate(photo_ids, start=1):
        photo = Photo.objects.filter(pk=photo_id).only("id", "data").first()
        if photo is None:  # deleted while we were working
            continue
        try:
            found = faces.find_faces(photo.data)
        except faces.FaceError as exc:
            if not faces.models_available():
                raise
            logger.info("Skipping photo %s: %s", photo_id, exc)
            found, unreadable = [], unreadable + 1
        _save_faces(photo_id, found)
        update(
            force=done == total, stage=f"Finding faces… photo {done:,} of {total:,}", scanned=done, unreadable=unreadable,
            percent=SCAN_SHARE * done // total,
        )


def _save_faces(photo_id, found):
    with transaction.atomic():
        # Another sync may have scanned it already; only the first one counts.
        if not Photo.objects.filter(pk=photo_id, faces_scanned=False).update(faces_scanned=True):
            return
        PhotoFace.objects.bulk_create([
            PhotoFace(
                photo_id=photo_id, x=face.x, y=face.y, width=face.width, height=face.height,
                confidence=face.confidence, embedding=faces.embedding_to_bytes(face.embedding),
            )
            for face in found
        ])


def score_photos(profile_embedding: np.ndarray) -> dict:
    """Score every scanned photo against the profile face and save each photo's best face."""
    best = {}  # photo id -> (similarity, face id)
    rows = PhotoFace.objects.values_list("id", "photo_id", "embedding").iterator(chunk_size=2000)
    for face_id, photo_id, embedding in rows:
        similarity = float(faces.embedding_from_bytes(embedding) @ profile_embedding)
        if photo_id not in best or similarity > best[photo_id][0]:
            best[photo_id] = (similarity, face_id)

    updates = [
        Photo(pk=photo_id, match_percent=faces.similarity_to_percent(similarity), matched_face_id=face_id)
        for photo_id, (similarity, face_id) in best.items()
    ]
    with transaction.atomic():
        clear_scores()
        Photo.objects.bulk_update(updates, ["match_percent", "matched_face"], batch_size=500)
    return {
        "total_photos": Photo.objects.count(),
        "photos_with_faces": len(best),
        "matched": sum(1 for photo in updates if photo.is_match),
    }


def _job_updater(job_id):
    state = {"written_at": 0.0}

    def update(force=False, **fields):
        now = time.monotonic()
        if not force and now - state["written_at"] < PROGRESS_INTERVAL:
            return
        state["written_at"] = now
        if "stage" in fields:
            fields["stage"] = fields["stage"][:255]
        MatchJob.objects.filter(pk=job_id).update(updated_at=timezone.now(), **fields)

    return update


def refresh_if_stale(job: MatchJob) -> MatchJob:
    """Mark a sync as failed when its thread is gone (e.g. the server was restarted)."""
    if not job.finished and (timezone.now() - job.updated_at).total_seconds() > STALE_AFTER_SECONDS:
        job.status = MatchJob.FAILED
        job.error = "The sync stopped unexpectedly (the server may have restarted). Please try again."
        job.save(update_fields=["status", "error", "updated_at"])
    return job
