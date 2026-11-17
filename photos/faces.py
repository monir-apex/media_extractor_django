"""Find faces in photos and compare them with the profile face.

Uses two small models from the OpenCV model zoo (both Apache-2.0, free for commercial use):
- YuNet finds faces and their landmarks (eyes, nose, mouth corners).
- SFace turns each face into 128 numbers (an "embedding"). Two photos of the same person
  give embeddings that point in a similar direction, so cosine similarity measures the match.

The model files are downloaded once into settings.FACE_MODELS_DIR.
"""
import hashlib
import io
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}

EMBEDDING_SIZE = 128
# Faces below this detector confidence are ignored.
DETECTION_CONFIDENCE = 0.8
# Faces narrower than this (in original image pixels) are too blurry to recognise reliably.
MIN_FACE_PIXELS = 16
# Small photos are enlarged before detection so small faces in group photos are found;
# large ones are shrunk to keep detection fast.
MIN_WORKING_SIDE = 1280
MAX_WORKING_SIDE = 1920
MAX_UPSCALE = 4

# SFace's published threshold: cosine similarity at or above this means "same person".
SAME_PERSON_SIMILARITY = 0.363
# Similarity that we show as 100%. Different photos of the same person rarely go above this.
FULL_MATCH_SIMILARITY = 0.8


class FaceError(Exception):
    """Problem with the face models or with reading an image."""


@dataclass
class Face:
    # Box as fractions (0-1) of the image width/height, so it can be drawn at any display size.
    x: float
    y: float
    width: float
    height: float
    confidence: float
    embedding: np.ndarray  # unit-length float32 vector

    @property
    def area(self):
        return self.width * self.height


def similarity_to_percent(similarity: float) -> int:
    """Turn cosine similarity into a 0-100 match score people can read.

    The scale is piecewise linear so that SFace's "same person" threshold lands exactly on
    settings.FACE_MATCH_PERCENT (70% by default): below it means "probably someone else".
    """
    match_percent = settings.FACE_MATCH_PERCENT
    if similarity <= 0:
        return 0
    if similarity < SAME_PERSON_SIMILARITY:
        # Stay strictly below the match line, even after rounding.
        return min(match_percent - 1, round(similarity / SAME_PERSON_SIMILARITY * match_percent))
    span = (similarity - SAME_PERSON_SIMILARITY) / (FULL_MATCH_SIMILARITY - SAME_PERSON_SIMILARITY)
    return min(100, round(match_percent + span * (100 - match_percent)))


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def embedding_from_bytes(data) -> np.ndarray:
    return np.frombuffer(bytes(data), dtype=np.float32)


# --- Models ---------------------------------------------------------------------------

def models_dir() -> Path:
    return Path(settings.FACE_MODELS_DIR)


def models_available() -> bool:
    return all((models_dir() / name).is_file() for name, _, _ in MODELS.values())


def ensure_models():
    """Download any missing model file, checking its SHA-256. Raises FaceError on failure."""
    folder = models_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for name, url, sha256 in MODELS.values():
        target = folder / name
        if target.is_file():
            continue
        logger.info("Downloading face model %s", name)
        partial = target.with_suffix(".part")
        digest = hashlib.sha256()
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(partial, "wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        digest.update(chunk)
                        out.write(chunk)
        except (requests.RequestException, OSError) as exc:
            partial.unlink(missing_ok=True)
            raise FaceError(f"Could not download the face model {name}: {exc}") from exc
        if digest.hexdigest() != sha256:
            partial.unlink(missing_ok=True)
            raise FaceError(f"The downloaded face model {name} is corrupt. Please try again.")
        partial.rename(target)


# OpenCV model objects are not thread-safe, so each thread gets its own pair.
_local = threading.local()


def _models():
    if getattr(_local, "models", None) is None:
        if not models_available():
            raise FaceError("The face models are not downloaded yet.")
        import cv2

        # OpenCV 5 warns on every model load that its new engine ignores target hints; that is harmless.
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        folder = models_dir()
        detector = cv2.FaceDetectorYN.create(
            str(folder / MODELS["detector"][0]), "", (320, 320), DETECTION_CONFIDENCE, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF.create(str(folder / MODELS["recognizer"][0]), "")
        _local.models = (detector, recognizer)
    return _local.models


# --- Images ---------------------------------------------------------------------------

def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into an OpenCV BGR array, upright according to its EXIF orientation.

    Browsers also apply EXIF orientation, so face boxes line up with what people see.
    """
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(bytes(data))) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.asarray(image)
    except Exception as exc:  # Pillow raises many different errors for bad files
        raise FaceError(f"Could not read the image: {exc}") from exc
    return np.ascontiguousarray(rgb[:, :, ::-1])


def find_faces(data: bytes) -> list:
    """Return every face found in the image, largest first."""
    import cv2

    image = decode_image(data)
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest < MIN_WORKING_SIDE:
        scale = min(MAX_UPSCALE, MIN_WORKING_SIDE / longest)
    else:
        scale = min(1.0, MAX_WORKING_SIDE / longest)
    if scale != 1.0:
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

    detector, recognizer = _models()
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, detections = detector.detect(image)
    faces = []
    for row in detections if detections is not None else []:
        x, y, w, h = (float(v) / scale for v in row[:4])
        if w < MIN_FACE_PIXELS:
            continue
        embedding = recognizer.feature(recognizer.alignCrop(image, row)).flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if not norm:
            continue
        # Clip the box to the image: faces at the edge can stick out a little.
        left, top = max(0.0, x), max(0.0, y)
        right, bottom = min(float(width), x + w), min(float(height), y + h)
        faces.append(Face(
            x=left / width, y=top / height,
            width=(right - left) / width, height=(bottom - top) / height,
            confidence=float(row[-1]), embedding=embedding / norm,
        ))
    faces.sort(key=lambda face: face.area, reverse=True)
    return faces


class ProfileFaceError(FaceError):
    """The profile photo has no usable face, or more than one similar-sized face."""


def profile_face(data: bytes) -> Face:
    """Pick the one face in a profile photo.

    When there are several faces the largest is used, but only if it is clearly the main one
    (at least twice the size of the next); otherwise we can't tell who the profile is.
    """
    faces = find_faces(data)
    if not faces:
        raise ProfileFaceError("No face was found in this photo. Please use a clear photo of your face.")
    if len(faces) > 1 and faces[0].area < 2 * faces[1].area:
        raise ProfileFaceError(
            f"This photo has {len(faces)} faces. Please use a photo where only your face is clearly visible."
        )
    return faces[0]
