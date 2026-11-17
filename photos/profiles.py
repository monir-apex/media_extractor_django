"""Validate profile photo uploads."""

MAX_PHOTO_BYTES = 5 * 1024 * 1024

# File signatures of the formats every browser can show in an <img>.
SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class InvalidPhoto(Exception):
    pass


def read_profile_photo(upload):
    """Return (bytes, content_type) for an uploaded photo, or raise InvalidPhoto."""
    if upload.size > MAX_PHOTO_BYTES:
        raise InvalidPhoto(f"The photo must be smaller than {MAX_PHOTO_BYTES // (1024 * 1024)} MB.")
    data = upload.read()
    content_type = sniff_image_type(data)
    if not content_type:
        raise InvalidPhoto("Please choose a JPEG, PNG, GIF or WebP image.")
    return data, content_type


def sniff_image_type(data: bytes):
    """Detect the image type from the file's contents, not its name or the browser's claim."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    for signature, content_type in SIGNATURES:
        if data.startswith(signature):
            return content_type
    return None
