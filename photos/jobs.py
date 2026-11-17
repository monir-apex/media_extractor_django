"""Run downloads in a background thread and record their progress in DownloadJob.

The page that started a download polls the job's status, so the user sees a
percentage and a photo count instead of a long, silent request. Threads live in
the web server process: a job still running when the server stops is marked as
failed the next time its status is read.
"""
import logging
import threading
import time

import requests
from django.db import close_old_connections, connection
from django.utils import timezone

from .downloaders import DownloadError, clean_limit, download_photos, validate_public_url
from .models import DownloadJob

logger = logging.getLogger(__name__)

# Write progress to the database at most this often (seconds).
PROGRESS_INTERVAL = 0.5
# A running job with no progress update for this long is assumed dead (server restarted).
STALE_AFTER_SECONDS = 60 * 60


def start_job(url, limit=None) -> DownloadJob:
    """Validate the request, then start downloading in the background.

    Raises DownloadError straight away for a bad link or limit, so the form can show it.
    """
    url = (url or "").strip()
    validate_public_url(url)
    limit = None if limit in (None, "") else clean_limit(limit)
    job = DownloadJob.objects.create(source_url=url, limit=limit)
    _launch(job.pk)
    return job


def _launch(job_id):
    threading.Thread(target=_run_in_thread, args=(job_id,), daemon=True, name=f"download-{job_id}").start()


def _run_in_thread(job_id):
    close_old_connections()
    try:
        run_job(job_id)
    finally:
        connection.close()


def run_job(job_id):
    job = DownloadJob.objects.get(pk=job_id)
    DownloadJob.objects.filter(pk=job_id).update(status=DownloadJob.RUNNING, updated_at=timezone.now())
    try:
        batch = download_photos(job.source_url, job.limit, on_progress=_progress_writer(job_id))
    except DownloadError as exc:
        _finish(job_id, status=DownloadJob.FAILED, error=str(exc), details=exc.details)
    except requests.RequestException as exc:
        _finish(job_id, status=DownloadJob.FAILED, error=f"Could not reach the link: {exc}")
    except Exception as exc:  # never leave a job stuck at "running"
        logger.exception("Download job %s failed", job_id)
        _finish(job_id, status=DownloadJob.FAILED, error=f"Unexpected error: {exc}")
    else:
        _finish(job_id, status=DownloadJob.DONE, batch=batch, saved=batch.photos.count(),
                details=batch.errors, stage="Done", percent=100)


def _finish(job_id, **fields):
    DownloadJob.objects.filter(pk=job_id).update(updated_at=timezone.now(), **fields)


def _progress_writer(job_id):
    state = {"written_at": 0.0, "percent": None}

    def write(result, force=False):
        now = time.monotonic()
        if not force and now - state["written_at"] < PROGRESS_INTERVAL:
            return
        state["written_at"] = now
        percent = result.percent
        # Totals can grow as more pages are found; never let the bar move backwards.
        if percent is not None and state["percent"] is not None:
            percent = max(percent, state["percent"])
        state["percent"] = percent if percent is not None else state["percent"]
        DownloadJob.objects.filter(pk=job_id).update(
            stage=result.stage[:255], found=result.found, checked=result.checked,
            saved=len(result.files), percent=state["percent"], updated_at=timezone.now(),
        )

    return write


def refresh_if_stale(job: DownloadJob) -> DownloadJob:
    """Mark a job as failed when its thread is gone (e.g. the server was restarted)."""
    if not job.finished and (timezone.now() - job.updated_at).total_seconds() > STALE_AFTER_SECONDS:
        job.status = DownloadJob.FAILED
        job.error = "The download stopped unexpectedly (the server may have restarted). Please try again."
        job.save(update_fields=["status", "error", "updated_at"])
    return job
