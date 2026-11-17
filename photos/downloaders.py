"""Download every photo behind a public share link and store it in the database.

Supported sources:
    * Google Drive  - shared folder or single file links
    * Dropbox       - shared folder (served as a zip) or single file links
    * Google Photos - shared album links
    * MEGA          - shared folder or file links (decrypted locally)
    * OneDrive      - personal shared folder or file links (1drv.ms, onedrive.live.com)
    * Anything else - the page is scraped for images, following its pagination
                      ("Next", page numbers) to every page; if the HTML has (almost) no
                      photos, it is loaded in headless Chromium to catch JavaScript-loaded
                      photos and JavaScript pagination ("Load more" buttons)
"""
import base64
import hashlib
import ipaddress
import mimetypes
import re
import shutil
import socket
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from django.conf import settings
from django.db import transaction

from . import browser, mega
from .models import DownloadBatch, Photo

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".heic", ".heif", ".avif",
    ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf",
}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
TIMEOUT = 30
CHUNK_SIZE = 1024 * 256
GDRIVE_FOLDER_LIMIT = 50
# OneDrive's own web viewer reads shared links with an anonymous "badger" token.
ONEDRIVE_TOKEN_URL = "https://api-badgerp.svc.ms/v1.0/token"
ONEDRIVE_APP_ID = "5cbed6ac-a083-4e14-b191-b4ba07653de2"
ONEDRIVE_API = "https://my.microsoftpersonalcontent.com/_api/v2.0"

# Not every Python version knows these, and they're needed for the stored content type.
for _type, _ext in (("image/heic", ".heic"), ("image/heif", ".heif"), ("image/avif", ".avif"), ("image/webp", ".webp")):
    mimetypes.add_type(_type, _ext)

MAX_PHOTOS = getattr(settings, "PHOTO_DOWNLOAD_MAX_PHOTOS", 100_000)
MAX_FILE_BYTES = getattr(settings, "PHOTO_DOWNLOAD_MAX_FILE_BYTES", 100 * 1024 * 1024)
MAX_ARCHIVE_BYTES = getattr(settings, "PHOTO_DOWNLOAD_MAX_ARCHIVE_BYTES", 2 * 1024 * 1024 * 1024)
# Scraped pages contain logos and icons; ignore images smaller than this.
MIN_SCRAPED_IMAGE_BYTES = getattr(settings, "PHOTO_DOWNLOAD_MIN_SCRAPED_BYTES", 10 * 1024)
# Web pages whose HTML yields fewer photos than this are re-loaded in a headless browser.
BROWSER_FALLBACK_BELOW = getattr(settings, "PHOTO_DOWNLOAD_BROWSER_FALLBACK_BELOW", 3)
# Most pages of a paginated gallery that are visited.
MAX_PAGES = getattr(settings, "PHOTO_DOWNLOAD_MAX_PAGES", 5_000)

# URLs that look like page N of a listing: ?page=2, ?p=2, ?offset=40, /page/2/, /page-2 ...
PAGE_URL_RE = re.compile(
    r"[?&;](?:page|pg|p|paged|pagenum|page_num|pageno|offset|start)=\d+"
    r"|/(?:page|pg|p)[/-]?\d+/?(?:$|[?#])",
    re.IGNORECASE,
)
# Link text of a "next page" control.
NEXT_TEXT_RE = re.compile(
    r"^(?:next(?: page)?|older(?: posts| entries)?|load more|show more|see more|view more|"
    r"more photos|›|»|>|→|>>)\s*[›»>→]*$",
    re.IGNORECASE,
)
NEXT_CLASS_RE = re.compile(r"(?:^|[\s_-])next(?:$|[\s_-])", re.IGNORECASE)


class DownloadError(Exception):
    """The link cannot be downloaded; the message is safe to show to the client."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or []


@dataclass
class DownloadResult:
    """Files a provider handler wrote into the temporary working folder."""

    url: str
    provider: str
    limit: int = MAX_PHOTOS
    files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    # Progress: ``found`` items queued for download, ``checked`` of them tried so far.
    found: int = 0
    checked: int = 0
    stage: str = ""
    on_progress: Optional[Callable[["DownloadResult", bool], None]] = field(default=None, repr=False)

    def add_found(self, count: int):
        self.found += count
        self._report()

    def mark_checked(self):
        self.checked += 1
        self._report()

    def set_stage(self, stage: str):
        self.stage = stage
        self._report(force=True)

    def _report(self, force: bool = False):
        if self.on_progress:
            self.on_progress(self, force)

    @property
    def percent(self) -> Optional[int]:
        """Rough completion (0-99) while running, or None while the total is still unknown.

        With a limit, progress is photos saved out of the limit. Without one it is images
        checked out of images found (some found images are skipped, e.g. tiny icons).
        """
        if self.limit < MAX_PHOTOS:
            fraction = len(self.files) / self.limit
        elif self.found:
            fraction = self.checked / self.found
        else:
            return None
        return min(int(fraction * 100), 99)


def download_photos(url: str, limit: Optional[int] = None, on_progress=None) -> DownloadBatch:
    """Download the photos behind ``url`` (at most ``limit``) and save them as one DownloadBatch.

    ``on_progress(result, force)`` is called as the download advances.
    """
    url = (url or "").strip()
    validate_public_url(url)
    provider = detect_provider(url)
    result = DownloadResult(url=url, provider=provider, limit=clean_limit(limit), on_progress=on_progress)
    result.set_stage("Reading the link…")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    # Handlers download to disk first (gdown and zip extraction need real files),
    # then everything is copied into the database and the folder is discarded.
    with tempfile.TemporaryDirectory(prefix="photo_download_") as tmp:
        dest = Path(tmp)
        HANDLERS[provider](session, url, dest, result)
        if not result.files:
            raise DownloadError("No photos could be downloaded from the link.", result.errors)
        result.set_stage(f"Saving {len(result.files)} photos…")
        return _save_to_db(result, dest)


@transaction.atomic
def _save_to_db(result: DownloadResult, dest: Path) -> DownloadBatch:
    batch = DownloadBatch.objects.create(
        source_url=result.url, provider=result.provider, errors=result.errors
    )
    seen = set()
    for relative in result.files:
        data = (dest / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        # Pages and albums often reference the same photo more than once.
        if digest in seen:
            continue
        seen.add(digest)
        name = Path(relative).name
        Photo.objects.create(
            batch=batch,
            filename=name,
            content_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
            size=len(data),
            sha256=digest,
            data=data,
        )
    return batch


def clean_limit(limit) -> int:
    """Validate a requested photo limit; empty means as many as allowed (MAX_PHOTOS)."""
    if limit in (None, ""):
        return MAX_PHOTOS
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise DownloadError("The limit must be a whole number.")
    if not 1 <= value <= MAX_PHOTOS:
        raise DownloadError(f"The limit must be between 1 and {MAX_PHOTOS}.")
    return value


def detect_provider(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host in ("drive.google.com", "docs.google.com"):
        return "google_drive"
    if host == "dropbox.com" or host.endswith(".dropbox.com"):
        return "dropbox"
    if host in ("photos.app.goo.gl", "photos.google.com", "goo.gl"):
        return "google_photos"
    if host in ("mega.nz", "mega.co.nz", "www.mega.nz"):
        return "mega"
    if host in ("1drv.ms", "onedrive.live.com"):
        return "onedrive"
    return "generic"


def validate_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts that resolve to private/internal addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise DownloadError("A valid http(s) URL is required.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise DownloadError(f"Could not resolve host '{parsed.hostname}'.")
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise DownloadError("URLs pointing to private or local addresses are not allowed.")


# --------------------------------------------------------------------------- providers


def _download_google_drive(session, url, dest, result):
    import gdown

    if "/folders/" not in url and "folderview" not in url:
        try:
            path = gdown.download(url=url, output=str(dest) + "/", quiet=True, fuzzy=True, use_cookies=False)
        except Exception as exc:
            raise DownloadError(f"Could not download the Google Drive file: {exc}")
        if not path:
            raise DownloadError("Could not download the Google Drive file. Is it shared as 'Anyone with the link'?")
        _keep_if_image(Path(path), dest, result)
        return

    try:
        entries = gdown.download_folder(
            url=url, output=str(dest), quiet=True, use_cookies=False,
            remaining_ok=True, skip_download=True,
        )
    except Exception as exc:
        raise DownloadError(f"Could not read the Google Drive folder: {exc}")
    if entries is None:
        raise DownloadError("Could not read the Google Drive folder. Is it shared as 'Anyone with the link'?")
    if len(entries) >= GDRIVE_FOLDER_LIMIT:
        result.errors.append(
            f"Google Drive only lists {GDRIVE_FOLDER_LIMIT} files per folder without an API key; "
            "some photos may have been missed."
        )

    images = [entry for entry in entries if is_image_name(entry.local_path)]
    result.set_stage("Downloading photos…")
    result.add_found(min(len(images), result.limit))
    for entry in images:
        if len(result.files) >= result.limit:
            result.errors.append(f"Stopped after {result.limit} photos.")
            break
        _download_drive_entry(gdown, entry, dest, result)
        result.mark_checked()


def _download_drive_entry(gdown, entry, dest, result):
    local_path = Path(entry.local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(id=entry.id, output=str(local_path), quiet=True, use_cookies=False)
    except Exception as exc:
        result.errors.append(f"{entry.path}: {exc}")
        return
    if local_path.exists():
        result.files.append(str(local_path.relative_to(dest)))
    else:
        result.errors.append(f"{entry.path}: download failed")


def _download_dropbox(session, url, dest, result):
    response = session.get(dropbox_direct_url(url), stream=True, timeout=TIMEOUT)
    if response.status_code != 200:
        raise DownloadError(f"Dropbox returned HTTP {response.status_code}. Is the link public?")

    content_type = response.headers.get("Content-Type", "")
    disposition = response.headers.get("Content-Disposition", "")
    if "zip" in content_type or ".zip" in disposition.lower():
        archive = dest / f"_{uuid.uuid4().hex}.zip"
        result.set_stage("Downloading the Dropbox folder (as one zip file)…")
        try:
            _save_stream(response, archive, MAX_ARCHIVE_BYTES)
            _extract_images_from_zip(archive, dest, result)
        finally:
            archive.unlink(missing_ok=True)
        return

    name = filename_from_response(response, url)
    if not (content_type.startswith("image/") or is_image_name(name)):
        raise DownloadError("The Dropbox link does not point to a photo or a folder.")
    path = _unique_path(dest, name)
    _save_stream(response, path, MAX_FILE_BYTES)
    result.files.append(path.name)


def _resolve_google_photos_short_link(url: str) -> str:
    """Turn a photos.app.goo.gl / goo.gl short link into the full photos.google.com URL.

    With a browser User-Agent, goo.gl serves a JavaScript redirect page that requests
    cannot follow. A plain client gets a normal HTTP redirect instead, so follow those.
    """
    if (urlparse(url).hostname or "").lower() not in ("photos.app.goo.gl", "goo.gl"):
        return url
    for _ in range(5):
        response = requests.get(url, timeout=TIMEOUT, allow_redirects=False)
        location = response.headers.get("Location")
        if not response.is_redirect or not location:
            break
        url = urljoin(url, location)
        if (urlparse(url).hostname or "").lower() == "photos.google.com":
            return url
    raise DownloadError("Could not open the Google Photos short link. Is the album shared?")


def _download_google_photos(session, url, dest, result):
    url = _resolve_google_photos_short_link(url)
    response = session.get(url, timeout=TIMEOUT)
    if response.status_code != 200:
        raise DownloadError(f"Google Photos returned HTTP {response.status_code}. Is the album shared?")

    photo_urls = extract_google_photos_urls(response.text)
    if not photo_urls:
        raise DownloadError("No photos found in the Google Photos album.")
    result.set_stage("Downloading photos…")
    result.add_found(min(len(photo_urls), result.limit))
    for index, photo_url in enumerate(photo_urls[:result.limit], start=1):
        # "=d" asks Google for the original file instead of a resized preview.
        _fetch_image(session, f"{photo_url}=d", dest, result, fallback_name=f"photo_{index:04d}.jpg")
        result.mark_checked()
    if len(photo_urls) > result.limit:
        result.errors.append(f"Stopped after {result.limit} photos.")


def _download_generic(session, url, dest, result):
    response = session.get(url, stream=True, timeout=TIMEOUT)
    if response.status_code != 200:
        raise DownloadError(f"The site returned HTTP {response.status_code}.")

    if response.headers.get("Content-Type", "").startswith("image/"):
        path = _unique_path(dest, filename_from_response(response, url))
        _save_stream(response, path, MAX_FILE_BYTES)
        result.files.append(path.name)
        return

    pending = []
    for page_number, page_image_urls in enumerate(_paginated_image_urls(session, response, result), start=1):
        pending += page_image_urls
        result.set_stage(f"Scanning page {page_number} · {result.found + len(pending)} images found")
        # Without a limit every page is scanned first, so the progress total is known.
        if len(pending) >= result.limit - len(result.files):
            _download_pending(session, pending, dest, result)
            if len(result.files) >= result.limit:
                break
    _download_pending(session, pending, dest, result)

    # Pages built with JavaScript show few or no photos in their HTML.
    if len(result.files) < min(BROWSER_FALLBACK_BELOW, result.limit):
        _download_with_browser(session, response.url, dest, result)
    if not result.files:
        raise DownloadError("No photos found on the page.", result.errors)


def _download_pending(session, pending, dest, result):
    if pending:
        result.set_stage("Downloading photos…")
        _fetch_scraped_images(session, pending, dest, result)
        pending.clear()


def _paginated_image_urls(session, first_response, result) -> Iterator[List[str]]:
    """Yield, page by page, new image URLs from the first page and every page its pagination leads to."""
    image_urls = set()
    visited = set()
    queue = [(first_response.url, first_response.text)]
    while queue:
        page_url, html = queue.pop(0)
        if _page_key(page_url) in visited:
            continue
        if len(visited) >= MAX_PAGES:
            result.errors.append(f"Stopped after {MAX_PAGES} pages.")
            break
        if html is None:
            page_url, html = _fetch_page(session, page_url, result)
            if html is None or _page_key(page_url) in visited:
                continue
        visited.add(_page_key(page_url))

        new_urls = [u for u in extract_image_urls(html, page_url) if u not in image_urls]
        image_urls.update(new_urls)
        for next_url in extract_pagination_urls(html, page_url):
            if _page_key(next_url) not in visited:
                queue.append((next_url, None))
        yield new_urls


def _fetch_page(session, url, result):
    """Return (final_url, html) for a pagination page, or (url, None) if it can't be read."""
    try:
        validate_public_url(url)
        response = session.get(url, timeout=TIMEOUT)
    except (DownloadError, requests.RequestException) as exc:
        result.errors.append(f"{url}: {exc}")
        return url, None
    if response.status_code != 200:
        result.errors.append(f"{url}: HTTP {response.status_code}")
        return url, None
    if response.headers.get("Content-Type", "").startswith("image/"):
        return url, None
    return response.url, response.text


def _page_key(url: str) -> str:
    """Normalise a page URL so the same page isn't visited twice."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment="", path=parsed.path.rstrip("/") or "/"))


def _download_with_browser(session, url, dest, result):
    result.set_stage("Opening the page in a browser to find more photos (this can take a minute)…")
    try:
        captured = browser.capture_images(url, USER_AGENT, MAX_FILE_BYTES, max_pages=MAX_PAGES)
    except browser.BrowserUnavailable as exc:
        result.errors.append(f"Browser fallback unavailable: {exc}")
        return
    except Exception as exc:
        result.errors.append(f"Browser could not load the page: {exc}")
        return
    if captured.hit_page_limit:
        result.errors.append(f"Stopped after {MAX_PAGES} pages.")

    result.set_stage("Saving photos found in the browser…")
    result.add_found(len(captured.images))
    for image in captured.images:
        if len(result.files) >= result.limit:
            result.errors.append(f"Stopped after {result.limit} photos.")
            return
        result.mark_checked()
        if len(image.body) < MIN_SCRAPED_IMAGE_BYTES:
            continue
        path = _unique_path(dest, name_from_url(image.url, image.content_type))
        path.write_bytes(image.body)
        result.files.append(path.name)

    # Images referenced on the page that the browser never loaded (e.g. off-screen).
    loaded = {image.url for image in captured.images}
    _fetch_scraped_images(session, [u for u in captured.dom_urls if u not in loaded], dest, result)


def _download_mega(session, url, dest, result):
    try:
        files = mega.list_files(mega.parse_link(url))
    except mega.MegaError as exc:
        raise DownloadError(str(exc))
    photos = [f for f in files if is_image_name(f.name)]
    if not photos:
        raise DownloadError("The MEGA link contains no photos.")
    result.set_stage("Downloading photos…")
    result.add_found(min(len(photos), result.limit))

    for file in photos:
        if len(result.files) >= result.limit:
            result.errors.append(f"Stopped after {result.limit} photos.")
            break
        result.mark_checked()
        if file.size > MAX_FILE_BYTES:
            result.errors.append(f"{file.name}: file too large, skipped.")
            continue
        path = _unique_path(dest, file.name)
        try:
            with open(path, "wb") as fh:
                for chunk in mega.iter_content(session, file, CHUNK_SIZE):
                    fh.write(chunk)
        except (mega.MegaError, requests.RequestException) as exc:
            path.unlink(missing_ok=True)
            result.errors.append(f"{file.name}: {exc}")
            continue
        result.files.append(path.name)


def _download_onedrive(session, url, dest, result):
    try:
        response = session.post(ONEDRIVE_TOKEN_URL, json={"appId": ONEDRIVE_APP_ID}, timeout=TIMEOUT)
        response.raise_for_status()
        token = response.json()["token"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise DownloadError(f"Could not connect to OneDrive: {exc}")
    headers = {"Authorization": f"Badger {token}", "Prefer": "autoredeem"}

    share = onedrive_share_id(url)
    response = session.get(f"{ONEDRIVE_API}/shares/{share}/driveitem?$expand=children", headers=headers, timeout=TIMEOUT)
    if response.status_code in (401, 403, 404):
        raise DownloadError("Could not open the OneDrive link. Is it shared as 'Anyone with the link'?")
    if response.status_code != 200:
        raise DownloadError(f"OneDrive returned HTTP {response.status_code}.")
    root = response.json()

    if "folder" not in root:
        if not is_image_name(root.get("name", "")):
            raise DownloadError("The OneDrive link does not point to a photo or a folder.")
        result.add_found(1)
        _fetch_onedrive_item(session, root, dest, result)
        return

    result.set_stage("Listing the OneDrive folder…")
    photos = list(_onedrive_photos(session, headers, root, result))
    if not photos:
        raise DownloadError("The OneDrive folder contains no photos.", result.errors)
    result.set_stage("Downloading photos…")
    result.add_found(min(len(photos), result.limit))
    for item in photos:
        if len(result.files) >= result.limit:
            result.errors.append(f"Stopped after {result.limit} photos.")
            break
        result.mark_checked()
        _fetch_onedrive_item(session, item, dest, result)


def _onedrive_photos(session, headers, folder, result) -> Iterator[dict]:
    """Every photo in a OneDrive folder and its subfolders, following the API's paging."""
    folders = [folder]
    found = 0
    while folders and found < result.limit:
        current = folders.pop(0)
        children = current.get("children")
        next_url = current.get("children@odata.nextLink")
        if children is None:
            # Subfolders are listed without their children.
            drive_id = current.get("parentReference", {}).get("driveId")
            next_url = f"{ONEDRIVE_API}/drives/{drive_id}/items/{current['id']}/children"
            children = []
        while True:
            for child in children:
                if "folder" in child:
                    folders.append(child)
                elif is_image_name(child.get("name", "")):
                    found += 1
                    yield child
            if not next_url or found >= result.limit:
                break
            try:
                response = session.get(next_url, headers=headers, timeout=TIMEOUT)
                response.raise_for_status()
                page = response.json()
            except (requests.RequestException, ValueError) as exc:
                result.errors.append(f"{current.get('name', 'folder')}: could not list files ({exc})")
                break
            children = page.get("value", [])
            next_url = page.get("@odata.nextLink")
            result.set_stage(f"Listing the OneDrive folder · {found} photos found")


def _fetch_onedrive_item(session, item, dest, result):
    name = item.get("name") or "photo.jpg"
    if item.get("size", 0) > MAX_FILE_BYTES:
        result.errors.append(f"{name}: file too large, skipped.")
        return
    download_url = item.get("@content.downloadUrl")
    if not download_url:
        result.errors.append(f"{name}: OneDrive gave no download link.")
        return
    try:
        response = session.get(download_url, stream=True, timeout=TIMEOUT)
    except requests.RequestException as exc:
        result.errors.append(f"{name}: {exc}")
        return
    with response:
        if response.status_code != 200:
            result.errors.append(f"{name}: HTTP {response.status_code}")
            return
        path = _unique_path(dest, name)
        try:
            _save_stream(response, path, MAX_FILE_BYTES)
        except DownloadError as exc:
            result.errors.append(f"{name}: {exc}")
            return
    result.files.append(path.name)


HANDLERS = {
    "google_drive": _download_google_drive,
    "dropbox": _download_dropbox,
    "google_photos": _download_google_photos,
    "mega": _download_mega,
    "onedrive": _download_onedrive,
    "generic": _download_generic,
}


# --------------------------------------------------------------------------- parsing helpers


def dropbox_direct_url(url: str) -> str:
    """Turn a Dropbox share link into a direct download link (dl=1)."""
    parsed = urlparse(url)
    query = [(k, v) for k, v in parse_qsl(parsed.query) if k not in ("dl", "raw")]
    query.append(("dl", "1"))
    return urlunparse(parsed._replace(query=urlencode(query)))


def onedrive_share_id(url: str) -> str:
    """The API id of a OneDrive share link ("u!" + unpadded base64url of the link).

    Links copied from the browser's address bar (onedrive.live.com/?redeem=...) carry
    the original 1drv.ms link base64-encoded in their "redeem" parameter.
    """
    redeem = dict(parse_qsl(urlparse(url).query)).get("redeem")
    if redeem:
        try:
            url = base64.urlsafe_b64decode(redeem + "=" * (-len(redeem) % 4)).decode()
        except (ValueError, UnicodeDecodeError):
            pass
    return "u!" + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


def extract_google_photos_urls(html: str) -> List[str]:
    found = re.findall(r"https://lh3\.googleusercontent\.com/pw/[A-Za-z0-9_\-]+", html)
    return list(dict.fromkeys(found))


def extract_image_urls(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    for meta in soup.find_all("meta", property=["og:image", "og:image:url"]):
        candidates.append(meta.get("content"))
    sources = [s for s in soup.find_all("source") if s.find_parent("picture")]
    for img in soup.find_all("img") + sources:
        srcset = img.get("srcset") or img.get("data-srcset")
        if srcset:
            # The last srcset entry is conventionally the largest.
            candidates.append(srcset.split(",")[-1].strip().split(" ")[0])
        for attr in ("data-original", "data-full", "data-src", "src"):
            if img.get(attr):
                candidates.append(img.get(attr))
                break
    for link in soup.find_all("a", href=True):
        if is_image_name(urlparse(link["href"]).path):
            candidates.append(link["href"])

    urls = []
    for candidate in candidates:
        if not candidate or candidate.startswith("data:"):
            continue
        absolute = urljoin(base_url, candidate.strip())
        if urlparse(absolute).scheme in ("http", "https"):
            urls.append(absolute)
    return list(dict.fromkeys(urls))


def extract_pagination_urls(html: str, base_url: str) -> List[str]:
    """Links to other pages of the same listing: rel=next, "Next" links and page numbers."""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    for tag in soup.find_all(["link", "a"], href=True):
        if "next" in [rel.lower() for rel in tag.get("rel") or []]:
            candidates.append(tag["href"])
    for link in soup.find_all("a", href=True):
        text = " ".join(link.get_text(" ", strip=True).split())
        label = link.get("aria-label") or link.get("title") or ""
        classes = " ".join(link.get("class") or [])
        href = urljoin(base_url, link["href"])
        is_next = (
            NEXT_TEXT_RE.match(text)
            or re.search(r"\bnext\b", label, re.IGNORECASE)
            or NEXT_CLASS_RE.search(classes)
        )
        # Page numbers count only when the link or its container looks like pagination.
        is_page_number = (text.isdigit() or re.search(r"\bpage\s*\d+", label, re.IGNORECASE)) and (
            PAGE_URL_RE.search(href) or _in_pagination_block(link)
        )
        if is_next or is_page_number:
            candidates.append(link["href"])

    base_host = (urlparse(base_url).hostname or "").lower().removeprefix("www.")
    current = _page_key(base_url)
    urls = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate.startswith(("#", "javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, candidate)
        parsed = urlparse(absolute)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if parsed.scheme not in ("http", "https") or host != base_host or is_image_name(parsed.path):
            continue
        if _page_key(absolute) != current:
            urls.append(urldefrag(absolute)[0])
    return list(dict.fromkeys(urls))


def _in_pagination_block(tag, depth: int = 4) -> bool:
    for parent in list(tag.parents)[:depth]:
        names = " ".join(parent.get("class") or []) + " " + (parent.get("id") or "") + " " + (parent.get("aria-label") or "")
        if re.search(r"pag(?:e|ination|ing|er)", names, re.IGNORECASE):
            return True
    return False


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def filename_from_response(response, url: str, fallback: str = "photo") -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r"filename\*=(?:UTF-8'')?([^;]+)", disposition) or re.search(
        r'filename="?([^";]+)"?', disposition
    )
    content_type = response.headers.get("Content-Type", "")
    if not match:
        return name_from_url(url, content_type, fallback)
    return _with_extension(requests.utils.unquote(match.group(1).strip().strip('"')) or fallback, content_type)


def name_from_url(url: str, content_type: str = "", fallback: str = "photo") -> str:
    name = requests.utils.unquote(Path(urlparse(url).path).name) or fallback
    return _with_extension(name, content_type)


def _with_extension(name: str, content_type: str) -> str:
    if not Path(name).suffix:
        name += mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
    return name


# --------------------------------------------------------------------------- file helpers


def _fetch_scraped_images(session, image_urls, dest, result):
    result.add_found(len(image_urls))
    for image_url in image_urls:
        if len(result.files) >= result.limit:
            result.errors.append(f"Stopped after {result.limit} photos.")
            break
        result.mark_checked()
        try:
            validate_public_url(image_url)
        except DownloadError as exc:
            result.errors.append(f"{image_url}: {exc}")
            continue
        _fetch_image(session, image_url, dest, result, min_bytes=MIN_SCRAPED_IMAGE_BYTES)


def _fetch_image(session, url, dest, result, fallback_name="photo", min_bytes=0):
    try:
        response = session.get(url, stream=True, timeout=TIMEOUT)
    except requests.RequestException as exc:
        result.errors.append(f"{url}: {exc}")
        return
    with response:
        if response.status_code != 200:
            result.errors.append(f"{url}: HTTP {response.status_code}")
            return
        content_type = response.headers.get("Content-Type", "")
        name = filename_from_response(response, url, fallback_name)
        if "svg" in content_type or not (content_type.startswith("image/") or is_image_name(name)):
            return
        path = _unique_path(dest, name)
        try:
            size = _save_stream(response, path, MAX_FILE_BYTES)
        except DownloadError as exc:
            result.errors.append(f"{url}: {exc}")
            return
    if size < min_bytes:
        path.unlink(missing_ok=True)
        return
    result.files.append(path.name)


def _save_stream(response, path: Path, max_bytes: int) -> int:
    size = 0
    try:
        with open(path, "wb") as fh:
            for chunk in response.iter_content(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise DownloadError(f"File is larger than {max_bytes // (1024 * 1024)} MB.")
                fh.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return size


def _extract_images_from_zip(archive: Path, dest: Path, result):
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        raise DownloadError("Dropbox returned an invalid zip archive.")
    with zf:
        images = [info for info in zf.infolist() if not info.is_dir() and is_image_name(info.filename)]
        result.set_stage("Unpacking photos…")
        result.add_found(min(len(images), result.limit))
        for info in images:
            if len(result.files) >= result.limit:
                result.errors.append(f"Stopped after {result.limit} photos.")
                break
            result.mark_checked()
            if info.file_size > MAX_FILE_BYTES:
                result.errors.append(f"{info.filename}: file too large, skipped.")
                continue
            # Only the base name is used, so archive paths can't escape dest.
            path = _unique_path(dest, info.filename)
            with zf.open(info) as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            result.files.append(path.name)


def _keep_if_image(path: Path, dest: Path, result):
    if is_image_name(path.name):
        result.files.append(str(path.relative_to(dest)))
    else:
        path.unlink(missing_ok=True)
        result.errors.append(f"{path.name} is not a photo, skipped.")


def _unique_path(dest: Path, name: str) -> Path:
    clean = re.sub(r"[^\w.\- ]", "_", Path(name.replace("\\", "/")).name).strip(" .") or "photo"
    stem, suffix = Path(clean).stem[:150], Path(clean).suffix
    path = dest / f"{stem}{suffix}"
    counter = 1
    while path.exists():
        path = dest / f"{stem}_{counter}{suffix}"
        counter += 1
    return path

