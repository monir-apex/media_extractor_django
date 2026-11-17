import base64
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest import mock

from Crypto.Cipher import AES
from Crypto.Util import Counter

from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from . import browser, downloaders, jobs, mega
from .downloaders import (
    DownloadError,
    detect_provider,
    dropbox_direct_url,
    extract_google_photos_urls,
    extract_image_urls,
    extract_pagination_urls,
    validate_public_url,
)
from .models import DownloadBatch, DownloadJob, Photo, Profile


class ProviderDetectionTests(SimpleTestCase):
    def test_detects_known_providers(self):
        cases = {
            "https://drive.google.com/drive/folders/abc?usp=sharing": "google_drive",
            "https://www.dropbox.com/scl/fo/abc/xyz?rlkey=k&dl=0": "dropbox",
            "https://photos.app.goo.gl/AbCdEf": "google_photos",
            "https://mega.nz/folder/Fqx02bIb#uW4k_gmy47f8utEM3kMZAQ": "mega",
            "https://1drv.ms/f/c/07ccfd6c99242069/IgDnBnfb?e=bYgAuw": "onedrive",
            "https://onedrive.live.com/?id=abc&redeem=aHR0cHM6Ly8xZHJ2Lm1z": "onedrive",
            "https://example.com/gallery": "generic",
        }
        for url, provider in cases.items():
            self.assertEqual(detect_provider(url), provider, url)


class ParsingTests(SimpleTestCase):
    def test_dropbox_direct_url_forces_download(self):
        self.assertEqual(
            dropbox_direct_url("https://www.dropbox.com/scl/fo/abc/xyz?rlkey=k&dl=0"),
            "https://www.dropbox.com/scl/fo/abc/xyz?rlkey=k&dl=1",
        )

    def test_extract_google_photos_urls_dedupes(self):
        html = (
            '"https://lh3.googleusercontent.com/pw/AAA_1",'
            '"https://lh3.googleusercontent.com/pw/AAA_1",'
            '"https://lh3.googleusercontent.com/a/avatar",'
            '"https://lh3.googleusercontent.com/pw/BBB-2"'
        )
        self.assertEqual(
            extract_google_photos_urls(html),
            ["https://lh3.googleusercontent.com/pw/AAA_1", "https://lh3.googleusercontent.com/pw/BBB-2"],
        )

    def test_extract_image_urls_from_html(self):
        html = """
            <meta property="og:image" content="https://cdn.example.com/cover.jpg">
            <img src="/small.jpg" srcset="/a-1x.jpg 1x, /a-2x.jpg 2x">
            <img data-src="lazy.png" src="data:image/gif;base64,AAAA">
            <a href="/full/photo.JPG">full</a>
            <a href="/about">about</a>
            <video><source src="/clip.webm"></video>
            <picture><source srcset="/pic.webp"><img src="/pic.jpg"></picture>
        """
        self.assertEqual(
            extract_image_urls(html, "https://example.com/gallery/"),
            [
                "https://cdn.example.com/cover.jpg",
                "https://example.com/a-2x.jpg",
                "https://example.com/small.jpg",
                "https://example.com/gallery/lazy.png",
                "https://example.com/pic.jpg",
                "https://example.com/pic.webp",
                "https://example.com/full/photo.JPG",
            ],
        )

    def test_rejects_non_http_and_private_urls(self):
        for url in ("ftp://example.com/a.jpg", "file:///etc/passwd", "http://127.0.0.1/", "http://localhost/"):
            with self.assertRaises(DownloadError, msg=url):
                validate_public_url(url)


class PaginationTests(TestCase):
    def test_finds_next_and_numbered_page_links(self):
        html = """
            <link rel="next" href="/gallery?page=2">
            <nav class="pagination">
              <a href="/gallery?page=1">1</a><a href="/gallery/page/3/">3</a>
              <a href="/gallery?page=2" aria-label="Next page">›</a>
            </nav>
            <a href="/blog/2024">2024</a>
            <a class="btn-next" href="/gallery?page=9">Older</a>
            <a href="https://other.com/gallery?page=2">Next</a>
            <a href="#top">Next</a>
        """
        self.assertEqual(
            extract_pagination_urls(html, "https://example.com/gallery?page=1"),
            ["https://example.com/gallery?page=2", "https://example.com/gallery/page/3/",
             "https://example.com/gallery?page=9"],
        )

    def test_downloads_photos_from_every_page(self):
        pages = {
            "https://example.com/gallery": '<img src="/1.jpg"><a href="?page=2">Next</a>',
            "https://example.com/gallery?page=2": '<img src="/2.jpg"><a href="?page=3">Next</a>'
                                                  '<a href="/gallery">1</a>',
            "https://example.com/gallery?page=3": '<img src="/3.jpg"><img src="/1.jpg">',
        }

        def fake_get(url, **kwargs):
            if url in pages:
                return mock.Mock(status_code=200, headers={"Content-Type": "text/html"}, text=pages[url], url=url)
            response = mock.MagicMock(status_code=200, headers={"Content-Type": "image/jpeg"})
            response.iter_content.return_value = [url.encode() * 1000]
            response.__enter__.return_value = response
            return response

        with mock.patch.object(downloaders, "validate_public_url"), \
                mock.patch("requests.Session.get", side_effect=fake_get), \
                mock.patch.object(browser, "capture_images") as capture:
            batch = downloaders.download_photos("https://example.com/gallery")

        capture.assert_not_called()
        self.assertEqual(sorted(p.filename for p in batch.photos.all()), ["1.jpg", "2.jpg", "3.jpg"])

    def test_photo_limit_stops_the_crawl(self):
        pages = {
            "https://example.com/gallery": '<img src="/1.jpg"><img src="/2.jpg"><a href="?page=2">Next</a>',
            "https://example.com/gallery?page=2": '<img src="/3.jpg"><a href="?page=3">Next</a>',
        }

        def fake_get(url, **kwargs):
            if url in pages:
                return mock.Mock(status_code=200, headers={"Content-Type": "text/html"}, text=pages[url], url=url)
            response = mock.MagicMock(status_code=200, headers={"Content-Type": "image/jpeg"})
            response.iter_content.return_value = [url.encode() * 1000]
            response.__enter__.return_value = response
            return response

        with mock.patch.object(downloaders, "validate_public_url"), \
                mock.patch("requests.Session.get", side_effect=fake_get) as get, \
                mock.patch.object(browser, "capture_images") as capture:
            batch = downloaders.download_photos("https://example.com/gallery", limit="2")

        capture.assert_not_called()
        self.assertEqual(sorted(p.filename for p in batch.photos.all()), ["1.jpg", "2.jpg"])
        self.assertNotIn("https://example.com/gallery?page=2", [c.args[0] for c in get.call_args_list])

    def test_invalid_limits_are_rejected(self):
        for limit in ("abc", "0", "-5", str(downloaders.MAX_PHOTOS + 1)):
            with self.assertRaises(DownloadError, msg=limit):
                downloaders.clean_limit(limit)
        self.assertEqual(downloaders.clean_limit(""), downloaders.MAX_PHOTOS)
        self.assertEqual(downloaders.clean_limit(None), downloaders.MAX_PHOTOS)
        self.assertEqual(downloaders.clean_limit("25"), 25)

    def test_page_limit_is_reported(self):
        def fake_get(url, **kwargs):
            number = int(url.rsplit("=", 1)[-1]) if "=" in url else 1
            html = f'<a href="/g?page={number + 1}">Next</a>'
            return mock.Mock(status_code=200, headers={"Content-Type": "text/html"}, text=html, url=url)

        result = downloaders.DownloadResult(url="https://example.com/g", provider="generic")
        session = mock.Mock(get=mock.Mock(side_effect=fake_get))
        with mock.patch.object(downloaders, "validate_public_url"), mock.patch.object(downloaders, "MAX_PAGES", 5):
            list(downloaders._paginated_image_urls(session, fake_get("https://example.com/g"), result))
        self.assertEqual(session.get.call_count, 4)
        self.assertEqual(result.errors, ["Stopped after 5 pages."])


class MegaTests(SimpleTestCase):
    def test_parse_link_formats(self):
        cases = {
            "https://mega.nz/folder/Fqx02bIb#uW4k_gmy47f8utEM3kMZAQ": ("folder", "Fqx02bIb", "uW4k_gmy47f8utEM3kMZAQ"),
            "https://mega.nz/folder/Fqx02bIb#uW4k_gmy47f8utEM3kMZAQ/folder/abc": ("folder", "Fqx02bIb", "uW4k_gmy47f8utEM3kMZAQ"),
            "https://mega.nz/file/AbC123#KeY-_9": ("file", "AbC123", "KeY-_9"),
            "https://mega.nz/#F!Fqx02bIb!uW4k_gmy47f8utEM3kMZAQ": ("folder", "Fqx02bIb", "uW4k_gmy47f8utEM3kMZAQ"),
            "https://mega.nz/#!AbC123!KeY": ("file", "AbC123", "KeY"),
        }
        for url, expected in cases.items():
            link = mega.parse_link(url)
            self.assertEqual((link.kind, link.id, link.key), expected, url)

    def test_link_without_key_is_rejected(self):
        with self.assertRaises(mega.MegaError):
            mega.parse_link("https://mega.nz/folder/Fqx02bIb")

    def test_folder_listing_is_decrypted(self):
        folder_key = bytes(range(16))
        file_key = bytes(range(100, 132))
        key, iv = mega._split_file_key(file_key)

        attributes = b'MEGA{"n":"beach.jpg"}'
        attributes += b"\0" * (-len(attributes) % 16)
        encrypted_attributes = AES.new(key, AES.MODE_CBC, iv=b"\0" * 16).encrypt(attributes)
        encrypted_key = AES.new(folder_key, AES.MODE_ECB).encrypt(file_key)

        def b64(data):
            return base64.urlsafe_b64encode(data).decode().rstrip("=")

        listing = {"f": [
            {"h": "folder1", "t": 1, "k": "x:" + b64(folder_key), "a": ""},
            {"h": "file1", "t": 0, "s": 123, "k": "folder1:" + b64(encrypted_key), "a": b64(encrypted_attributes)},
        ]}
        with mock.patch.object(mega, "_api", return_value=listing):
            files = mega.list_files(mega.MegaLink("folder", "Fqx02bIb", b64(folder_key)))

        self.assertEqual(len(files), 1)
        self.assertEqual((files[0].name, files[0].size, files[0].key, files[0].iv), ("beach.jpg", 123, key, iv))

    def test_file_contents_are_decrypted(self):
        key, iv = mega._split_file_key(bytes(range(32)))
        plain = b"\x89PNG fake image data" * 50
        cipher = AES.new(key, AES.MODE_CTR, counter=Counter.new(64, prefix=iv, initial_value=0))
        encrypted = cipher.encrypt(plain)

        session = mock.MagicMock()
        response = session.get.return_value.__enter__.return_value
        response.status_code = 200
        response.iter_content.return_value = [encrypted[:100], encrypted[100:]]
        file = mega.MegaFile(handle="h", name="a.png", size=len(plain), key=key, iv=iv, folder_id="f")

        with mock.patch.object(mega, "_api", return_value={"g": "https://example.com/dl"}):
            self.assertEqual(b"".join(mega.iter_content(session, file, 64)), plain)


class BrowserFallbackTests(TestCase):
    def test_javascript_page_falls_back_to_browser(self):
        page = mock.Mock(status_code=200, headers={"Content-Type": "text/html"}, text="<div id=app></div>",
                         url="https://example.com/gallery")
        captured = browser.BrowserResult(images=[
            browser.CapturedImage("https://cdn.example.com/p/1.jpg", "image/jpeg", b"a" * 20_000),
            browser.CapturedImage("https://cdn.example.com/icon", "image/png", b"tiny"),
        ])
        with mock.patch.object(downloaders, "validate_public_url"), \
                mock.patch("requests.Session.get", return_value=page), \
                mock.patch.object(browser, "capture_images", return_value=captured) as capture:
            batch = downloaders.download_photos("https://example.com/gallery")

        capture.assert_called_once()
        self.assertEqual([p.filename for p in batch.photos.all()], ["1.jpg"])  # icon below size limit

    def test_missing_browser_is_reported(self):
        page = mock.Mock(status_code=200, headers={"Content-Type": "text/html"}, text="<p>nothing</p>",
                         url="https://example.com/")
        with mock.patch.object(downloaders, "validate_public_url"), \
                mock.patch("requests.Session.get", return_value=page), \
                mock.patch.object(browser, "capture_images", side_effect=browser.BrowserUnavailable("not installed")):
            with self.assertRaises(DownloadError) as ctx:
                downloaders.download_photos("https://example.com/")
        self.assertIn("Browser fallback unavailable: not installed", ctx.exception.details)


class DownloadEndpointTests(TestCase):
    endpoint = "/api/photos/download/"

    def post(self, body):
        return self.client.post(self.endpoint, data=json.dumps(body), content_type="application/json")

    def fake_download(self, files):
        """Patch the generic handler to 'download' the given {name: bytes} files."""
        def handler(session, url, dest, result):
            for name, data in files.items():
                (dest / name).write_bytes(data)
                result.files.append(name)
            result.errors.append("https://example.com/broken.jpg: HTTP 404")

        return mock.patch.multiple(
            downloaders,
            validate_public_url=mock.DEFAULT,
            HANDLERS={"generic": handler},
        )

    def test_requires_url(self):
        self.assertEqual(self.post({}).status_code, 400)

    def test_rejects_get(self):
        self.assertEqual(self.client.get(self.endpoint).status_code, 405)

    def test_rejects_private_url(self):
        self.assertEqual(self.post({"url": "http://127.0.0.1:8000/"}).status_code, 400)

    def test_saves_photos_into_database(self):
        with self.fake_download({"a.jpg": b"jpeg-a", "b.png": b"png-b", "a_copy.jpg": b"jpeg-a"}):
            response = self.post({"url": "https://example.com/gallery"})
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["provider"], "generic")
        self.assertEqual([p["filename"] for p in data["photos"]], ["a.jpg", "b.png"])  # duplicate dropped
        self.assertEqual(data["errors"], ["https://example.com/broken.jpg: HTTP 404"])

        batch = DownloadBatch.objects.get(pk=data["batch_id"])
        photo = batch.photos.get(filename="b.png")
        self.assertEqual(bytes(photo.data), b"png-b")
        self.assertEqual(photo.content_type, "image/png")
        self.assertEqual(photo.size, 5)

    def test_no_photos_returns_422_and_saves_nothing(self):
        with self.fake_download({}):
            response = self.post({"url": "https://example.com/gallery"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["details"], ["https://example.com/broken.jpg: HTTP 404"])
        self.assertFalse(DownloadBatch.objects.exists())


class GalleryTests(TestCase):
    def setUp(self):
        batch = DownloadBatch.objects.create(source_url="https://example.com/a", provider="dropbox")
        self.photo = Photo.objects.create(
            batch=batch, filename="cat.jpg", content_type="image/jpeg", size=4, sha256="x", data=b"jpeg"
        )
        Photo.objects.create(
            batch=batch, filename="raw.heic", content_type="image/heic", size=4, sha256="y", data=b"heic"
        )

    def test_gallery_lists_photos(self):
        response = self.client.get("/photos/")
        self.assertContains(response, "cat.jpg")
        self.assertContains(response, "Dropbox")
        self.assertContains(response, f'src="/photos/{self.photo.pk}/image/"')
        self.assertContains(response, "no browser preview")

    def test_root_redirects_to_gallery(self):
        self.assertRedirects(self.client.get("/"), "/photos/")

    def test_image_is_served_from_database(self):
        response = self.client.get(f"/photos/{self.photo.pk}/image/")
        self.assertEqual(response.content, b"jpeg")
        self.assertEqual(response["Content-Type"], "image/jpeg")
        self.assertIn("inline", response["Content-Disposition"])

        response = self.client.get(f"/photos/{self.photo.pk}/image/?download=1")
        self.assertIn("attachment", response["Content-Disposition"])

    def test_delete_all_removes_every_photo(self):
        self.assertContains(self.client.get("/photos/"), "Delete all photos")
        response = self.client.post("/photos/delete-all/", follow=True)
        self.assertContains(response, "Deleted 2 photos.")
        self.assertFalse(Photo.objects.exists())
        self.assertFalse(DownloadBatch.objects.exists())
        self.assertNotContains(response, "Delete all photos")

    def test_delete_all_requires_post(self):
        self.assertEqual(self.client.get("/photos/delete-all/").status_code, 405)
        self.assertEqual(Photo.objects.count(), 2)

    def test_gallery_links_to_add_page(self):
        self.assertContains(self.client.get("/photos/"), 'href="/photos/add/"')


class AddLinkPageTests(TestCase):
    url = "/photos/add/"

    def test_shows_form(self):
        response = self.client.get(self.url)
        self.assertContains(response, 'name="url"')
        self.assertContains(response, 'name="limit"')
        self.assertContains(response, "csrfmiddlewaretoken")

    def test_error_keeps_url_in_form(self):
        response = self.client.post(self.url, {"url": "http://127.0.0.1/"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "private or local addresses")
        self.assertContains(response, 'value="http://127.0.0.1/"')

    def test_invalid_limit_shows_error_without_starting(self):
        with mock.patch.object(jobs, "_launch") as launch:
            response = self.client.post(self.url, {"url": "https://example.com/", "limit": "0"})
        self.assertContains(response, "The limit must be between 1 and")
        launch.assert_not_called()
        self.assertFalse(DownloadJob.objects.exists())

    def test_form_is_prefilled_from_query(self):
        response = self.client.get(self.url + "?url=https://example.com/a&limit=7")
        self.assertContains(response, 'value="https://example.com/a"')
        self.assertContains(response, 'value="7"')

    def test_success_redirects_to_new_batch(self):
        def handler(session, url, dest, result):
            (dest / "a.jpg").write_bytes(b"jpeg")
            result.files.append("a.jpg")

        # Run the "background" download right away so the test can check the outcome.
        with mock.patch.multiple(downloaders, validate_public_url=mock.DEFAULT, HANDLERS={"generic": handler}), \
                mock.patch.object(jobs, "validate_public_url"), \
                mock.patch.object(jobs, "_launch", side_effect=jobs.run_job):
            response = self.client.post(self.url, {"url": "https://example.com/gallery"})

        job = DownloadJob.objects.get()
        self.assertRedirects(response, f"/photos/jobs/{job.pk}/", fetch_redirect_response=False)
        status = self.client.get(f"/api/photos/jobs/{job.pk}/").json()
        self.assertEqual((status["status"], status["saved"], status["percent"]), ("done", 1, 100))

        batch = DownloadBatch.objects.get()
        response = self.client.get(f"/photos/jobs/{job.pk}/")
        self.assertRedirects(response, f"/photos/#batch-{batch.pk}", fetch_redirect_response=False)
        page = self.client.get("/photos/")
        self.assertContains(page, "Downloaded 1 photo from https://example.com/gallery")
        self.assertContains(page, "a.jpg")


class DownloadJobTests(TestCase):
    def test_progress_page_shows_percentage_and_count(self):
        job = DownloadJob.objects.create(
            source_url="https://example.com/g", limit=50, status=DownloadJob.RUNNING,
            stage="Downloading photos…", found=80, checked=30, saved=20, percent=40,
        )
        response = self.client.get(f"/photos/jobs/{job.pk}/")
        self.assertContains(response, "40%")
        self.assertContains(response, "20 of 50 photos downloaded")
        self.assertContains(response, "Checked 30 of 80 images")
        self.assertContains(response, f"/api/photos/jobs/{job.pk}/")

    def test_failed_job_shows_error_and_retry_link(self):
        job = DownloadJob.objects.create(
            source_url="https://example.com/g", limit=5, status=DownloadJob.FAILED,
            error="No photos found on the page.", details=["https://example.com/x.jpg: HTTP 404"],
        )
        response = self.client.get(f"/photos/jobs/{job.pk}/")
        self.assertContains(response, "Download failed")
        self.assertContains(response, "No photos found on the page.")
        self.assertContains(response, "/photos/add/?url=https%3A%2F%2Fexample.com%2Fg&amp;limit=5")

    def test_failure_is_recorded(self):
        job = DownloadJob.objects.create(source_url="https://example.com/g")
        with mock.patch.object(jobs, "download_photos", side_effect=DownloadError("Nope", ["detail"])):
            jobs.run_job(job.pk)
        job.refresh_from_db()
        self.assertEqual((job.status, job.error, job.details), ("failed", "Nope", ["detail"]))

    def test_progress_is_written_while_downloading(self):
        job = DownloadJob.objects.create(source_url="https://example.com/g", limit=4)
        seen = []

        def fake_download(url, limit, on_progress):
            result = downloaders.DownloadResult(url=url, provider="generic", limit=limit, on_progress=on_progress)
            result.set_stage("Downloading photos…")
            result.add_found(4)
            result.files += ["a.jpg", "b.jpg"]
            result.checked = 2
            result.set_stage("Downloading photos…")
            seen.append(DownloadJob.objects.values("saved", "percent", "found").get(pk=job.pk))
            raise DownloadError("stop here")

        with mock.patch.object(jobs, "download_photos", side_effect=fake_download):
            jobs.run_job(job.pk)
        self.assertEqual(seen, [{"saved": 2, "percent": 50, "found": 4}])  # 2 of limit 4

    def test_percent(self):
        result = downloaders.DownloadResult(url="", provider="generic")
        self.assertIsNone(result.percent)  # total unknown yet
        result.found, result.checked = 200, 50
        self.assertEqual(result.percent, 25)
        result.checked = 200
        self.assertEqual(result.percent, 99)  # 100% only once saved
        limited = downloaders.DownloadResult(url="", provider="generic", limit=10, files=["a"] * 6)
        limited.found, limited.checked = 20, 19  # with a limit, only saved photos count
        self.assertEqual(limited.percent, 60)

    def test_stale_job_is_marked_failed(self):
        job = DownloadJob.objects.create(source_url="https://example.com/g", status=DownloadJob.RUNNING)
        DownloadJob.objects.filter(pk=job.pk).update(updated_at=timezone.now() - timedelta(hours=2))
        status = self.client.get(f"/api/photos/jobs/{job.pk}/").json()
        self.assertEqual(status["status"], "failed")
        self.assertIn("stopped unexpectedly", status["error"])


class ProfileTests(TestCase):
    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    def upload(self, name, data, content_type="image/png"):
        from django.core.files.uploadedfile import SimpleUploadedFile
        return SimpleUploadedFile(name, data, content_type=content_type)

    def test_profile_page_shows_form(self):
        response = self.client.get("/profile/")
        self.assertContains(response, 'name="photo"')
        self.assertContains(response, 'name="display_name"')
        self.assertContains(response, 'enctype="multipart/form-data"')

    def test_upload_photo_and_name(self):
        response = self.client.post("/profile/", {"display_name": "Monir Hossain", "photo": self.upload("me.png", self.PNG)})
        self.assertRedirects(response, "/profile/")
        profile = Profile.objects.get()
        self.assertEqual((profile.display_name, profile.photo_content_type), ("Monir Hossain", "image/png"))

        photo = self.client.get("/profile/photo/")
        self.assertEqual(photo.content, self.PNG)
        self.assertEqual(photo["Content-Type"], "image/png")
        self.assertEqual(photo["X-Content-Type-Options"], "nosniff")

        gallery = self.client.get("/photos/")
        self.assertContains(gallery, 'href="/profile/"')
        self.assertContains(gallery, profile.get_photo_url())
        self.assertContains(gallery, "Monir Hossain")

    def test_saving_name_keeps_existing_photo(self):
        self.client.post("/profile/", {"display_name": "A", "photo": self.upload("me.png", self.PNG)})
        self.client.post("/profile/", {"display_name": "B"})
        profile = Profile.objects.get()
        self.assertEqual((profile.display_name, bytes(profile.photo)), ("B", self.PNG))

    def test_rejects_files_that_are_not_images(self):
        fake = self.upload("evil.png", b"<html><script>alert(1)</script>", content_type="image/png")
        response = self.client.post("/profile/", {"display_name": "A", "photo": fake})
        self.assertContains(response, "Please choose a JPEG, PNG, GIF or WebP image.")
        self.assertContains(response, 'value="A"')
        self.assertFalse(Profile.objects.get().has_photo)

    def test_rejects_large_photos(self):
        big = self.upload("big.png", self.PNG + b"\x00" * (5 * 1024 * 1024))
        response = self.client.post("/profile/", {"photo": big})
        self.assertContains(response, "smaller than 5 MB")

    def test_remove_photo(self):
        self.client.post("/profile/", {"display_name": "A", "photo": self.upload("me.png", self.PNG)})
        self.client.post("/profile/", {"action": "remove-photo"})
        self.assertFalse(Profile.objects.get().has_photo)
        self.assertEqual(self.client.get("/profile/photo/").status_code, 404)

    def test_initials_shown_without_photo(self):
        Profile.objects.create(pk=1, display_name="monir hossain noyon")
        self.assertContains(self.client.get("/photos/"), ">MH</span>")

    def test_sniffs_webp_and_jpeg(self):
        from .profiles import sniff_image_type
        self.assertEqual(sniff_image_type(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "image/webp")
        self.assertEqual(sniff_image_type(b"\xff\xd8\xff\xe0rest"), "image/jpeg")
        self.assertIsNone(sniff_image_type(b"GIF"))


class OneDriveTests(SimpleTestCase):
    LINK = "https://1drv.ms/f/c/07ccfd6c99242069/IgDnBnfb?e=bYgAuw"

    def test_share_id_encodes_link(self):
        expected = "u!" + base64.urlsafe_b64encode(self.LINK.encode()).decode().rstrip("=")
        self.assertEqual(downloaders.onedrive_share_id(self.LINK), expected)

    def test_share_id_uses_redeem_parameter(self):
        redeem = base64.b64encode(self.LINK.encode()).decode().rstrip("=")
        url = f"https://onedrive.live.com/?id=%2Fpersonal%2Fx&redeem={redeem}&ga=1"
        self.assertEqual(downloaders.onedrive_share_id(url), downloaders.onedrive_share_id(self.LINK))

    def _session(self, responses):
        def fake_get(url, **kwargs):
            if url.startswith("https://dl.example/"):
                response = mock.MagicMock(status_code=200)
                response.iter_content.return_value = [b"jpeg-" + url.encode()]
                response.__enter__.return_value = response
                return response
            return mock.Mock(status_code=200, json=mock.Mock(return_value=responses[url]),
                             raise_for_status=mock.Mock())

        token = mock.Mock(json=mock.Mock(return_value={"token": "t"}), raise_for_status=mock.Mock())
        return mock.Mock(post=mock.Mock(return_value=token), get=mock.Mock(side_effect=fake_get))

    def _photo(self, name):
        return {"name": name, "size": 10, "file": {}, "@content.downloadUrl": f"https://dl.example/{name}"}

    def test_downloads_photos_across_pages_and_subfolders(self):
        api = downloaders.ONEDRIVE_API
        root_url = f"{api}/shares/{downloaders.onedrive_share_id(self.LINK)}/driveitem?$expand=children"
        responses = {
            root_url: {
                "name": "Pictures", "folder": {},
                "children": [self._photo("a.jpg"), {"name": "notes.txt", "file": {}},
                             {"name": "Sub", "id": "sub1", "folder": {}, "parentReference": {"driveId": "d1"}}],
                "children@odata.nextLink": "https://next/page2",
            },
            "https://next/page2": {"value": [self._photo("b.png")]},
            f"{api}/drives/d1/items/sub1/children": {"value": [self._photo("c.jpg")]},
        }
        session = self._session(responses)
        with tempfile.TemporaryDirectory() as tmp:
            result = downloaders.DownloadResult(url=self.LINK, provider="onedrive")
            downloaders._download_onedrive(session, self.LINK, Path(tmp), result)
            self.assertEqual(sorted(result.files), ["a.jpg", "b.png", "c.jpg"])
            self.assertEqual((Path(tmp) / "a.jpg").read_bytes(), b"jpeg-https://dl.example/a.jpg")
        self.assertEqual(session.get.call_args_list[0].kwargs["headers"]["Authorization"], "Badger t")

    def test_limit_stops_listing(self):
        root_url = f"{downloaders.ONEDRIVE_API}/shares/{downloaders.onedrive_share_id(self.LINK)}/driveitem?$expand=children"
        responses = {root_url: {"name": "P", "folder": {}, "children": [self._photo("a.jpg"), self._photo("b.jpg")],
                                "children@odata.nextLink": "https://next/never"}}
        with tempfile.TemporaryDirectory() as tmp:
            result = downloaders.DownloadResult(url=self.LINK, provider="onedrive", limit=1)
            downloaders._download_onedrive(self._session(responses), self.LINK, Path(tmp), result)
        self.assertEqual(result.files, ["a.jpg"])

    def test_private_link_is_reported(self):
        session = self._session({})
        session.get = mock.Mock(return_value=mock.Mock(status_code=401))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(DownloadError, "Anyone with the link"):
            downloaders._download_onedrive(session, self.LINK, Path(tmp),
                                           downloaders.DownloadResult(url=self.LINK, provider="onedrive"))
