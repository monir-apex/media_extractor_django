import uuid

from django.conf import settings
from django.db import models
from django.urls import reverse


class DownloadBatch(models.Model):
    """All photos downloaded from one share link."""

    PROVIDER_CHOICES = [
        ('google_drive', 'Google Drive'),
        ('dropbox', 'Dropbox'),
        ('google_photos', 'Google Photos'),
        ('mega', 'MEGA'),
        ('onedrive', 'OneDrive'),
        ('generic', 'Web page'),
    ]

    source_url = models.URLField(max_length=2000)
    provider = models.CharField(max_length=32, choices=PROVIDER_CHOICES)
    errors = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'download batches'

    def __str__(self):
        return f'{self.provider}: {self.source_url}'


class Photo(models.Model):
    # Formats browsers can render in an <img> tag; others (HEIC, RAW) get a placeholder.
    PREVIEWABLE_TYPES = {
        'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/avif', 'image/bmp',
    }

    batch = models.ForeignKey(DownloadBatch, on_delete=models.CASCADE, related_name='photos')
    filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100)
    size = models.PositiveBigIntegerField()
    sha256 = models.CharField(max_length=64, db_index=True)
    data = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)
    # Face matching: set by a face-match sync (see photos/matching.py).
    faces_scanned = models.BooleanField(default=False)
    match_percent = models.PositiveSmallIntegerField(null=True, blank=True, db_index=True)
    matched_face = models.ForeignKey('PhotoFace', null=True, blank=True, on_delete=models.SET_NULL, related_name='+')

    class Meta:
        ordering = ['id']

    def __str__(self):
        return self.filename

    @property
    def previewable(self):
        return self.content_type in self.PREVIEWABLE_TYPES

    def get_absolute_url(self):
        return reverse('photo-image', args=[self.pk])

    @property
    def is_match(self):
        return self.match_percent is not None and self.match_percent >= settings.FACE_MATCH_PERCENT


class PhotoFace(models.Model):
    """One face found in a photo. The box is stored as fractions (0-1) of the image size."""

    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, related_name='faces')
    x = models.FloatField()
    y = models.FloatField()
    width = models.FloatField()
    height = models.FloatField()
    confidence = models.FloatField()
    embedding = models.BinaryField()  # 128 float32 values (see photos/faces.py)

    class Meta:
        ordering = ['id']

    def __str__(self):
        return f'Face {self.pk} in photo {self.photo_id}'

    @property
    def box_percent(self):
        """'left top width height' in percent, for drawing the box over the image."""
        return ' '.join(f'{value * 100:.2f}' for value in (self.x, self.y, self.width, self.height))

    @property
    def center_percent(self):
        """CSS object-position that keeps this face visible in a cropped thumbnail."""
        return f'{(self.x + self.width / 2) * 100:.1f}% {(self.y + self.height / 2) * 100:.1f}%'


class DownloadJob(models.Model):
    """One download running in the background, so the browser can show its progress."""

    QUEUED, RUNNING, DONE, FAILED = 'queued', 'running', 'done', 'failed'
    STATUS_CHOICES = [(QUEUED, 'Queued'), (RUNNING, 'Running'), (DONE, 'Done'), (FAILED, 'Failed')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source_url = models.URLField(max_length=2000)
    limit = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=QUEUED)
    stage = models.CharField(max_length=255, blank=True)
    found = models.PositiveIntegerField(default=0)
    checked = models.PositiveIntegerField(default=0)
    saved = models.PositiveIntegerField(default=0)
    percent = models.PositiveSmallIntegerField(null=True, blank=True)
    error = models.TextField(blank=True)
    details = models.JSONField(default=list, blank=True)
    batch = models.ForeignKey(DownloadBatch, null=True, blank=True, on_delete=models.SET_NULL, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.status}: {self.source_url}'

    @property
    def finished(self):
        return self.status in (self.DONE, self.FAILED)

    def as_dict(self):
        return {
            'id': str(self.pk),
            'url': self.source_url,
            'limit': self.limit,
            'status': self.status,
            'stage': self.stage,
            'found': self.found,
            'checked': self.checked,
            'saved': self.saved,
            'percent': self.percent,
            'error': self.error,
            'details': self.details,
            'batch_id': self.batch_id,
        }


class Profile(models.Model):
    """The app's single profile: a display name and a profile photo."""

    display_name = models.CharField(max_length=80, blank=True)
    photo = models.BinaryField(null=True, blank=True)
    photo_content_type = models.CharField(max_length=100, blank=True)
    # The profile face's embedding; empty until it has been computed (see photos/matching.py).
    face_embedding = models.BinaryField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.display_name or 'Profile'

    @classmethod
    def load(cls):
        profile, _ = cls.objects.get_or_create(pk=1)
        return profile

    @property
    def has_photo(self):
        return bool(self.photo_content_type)

    @property
    def initials(self):
        return ''.join(word[0] for word in self.display_name.split()[:2]).upper() or '?'

    def get_photo_url(self):
        # The version in the URL lets browsers cache the photo until it changes.
        return f"{reverse('profile-photo')}?v={int(self.updated_at.timestamp())}"


class MatchJob(models.Model):
    """One face-match sync running in the background, so the gallery can show its progress."""

    QUEUED, RUNNING, DONE, FAILED = 'queued', 'running', 'done', 'failed'
    STATUS_CHOICES = [(QUEUED, 'Queued'), (RUNNING, 'Running'), (DONE, 'Done'), (FAILED, 'Failed')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=QUEUED)
    stage = models.CharField(max_length=255, blank=True)
    to_scan = models.PositiveIntegerField(default=0)
    scanned = models.PositiveIntegerField(default=0)
    unreadable = models.PositiveIntegerField(default=0)
    total_photos = models.PositiveIntegerField(default=0)
    photos_with_faces = models.PositiveIntegerField(default=0)
    matched = models.PositiveIntegerField(default=0)
    percent = models.PositiveSmallIntegerField(null=True, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.status}: face match {self.created_at:%Y-%m-%d %H:%M}'

    @property
    def finished(self):
        return self.status in (self.DONE, self.FAILED)

    def as_dict(self):
        return {
            'id': str(self.pk),
            'status': self.status,
            'stage': self.stage,
            'to_scan': self.to_scan,
            'scanned': self.scanned,
            'unreadable': self.unreadable,
            'total_photos': self.total_photos,
            'photos_with_faces': self.photos_with_faces,
            'matched': self.matched,
            'percent': self.percent,
            'error': self.error,
        }
