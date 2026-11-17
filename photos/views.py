import json
from urllib.parse import urlencode

import requests
from django.contrib import messages
from django.core.paginator import Paginator
from django.conf import settings
from django.db.models import Count, Prefetch, Sum
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import faces, jobs, matching
from .downloaders import MAX_PHOTOS, DownloadError, download_photos
from .models import DownloadBatch, DownloadJob, MatchJob, Photo, Profile
from .profiles import InvalidPhoto, read_profile_photo


@csrf_exempt
@require_POST
def download_from_link(request):
    """Download every photo behind a public share link into the database.

    Body: {"url": "<Google Drive / Dropbox / Google Photos / any page link>", "limit": 100}
    ("limit" is optional: the most photos to download; leave it out for all of them.)
    """
    if request.content_type == 'application/json':
        try:
            payload = json.loads(request.body or b'{}')
        except ValueError:
            return JsonResponse({'error': 'Invalid JSON body.'}, status=400)
        payload = payload if isinstance(payload, dict) else {}
        url, limit = payload.get('url'), payload.get('limit')
    else:
        url, limit = request.POST.get('url'), request.POST.get('limit')

    if not url:
        return JsonResponse({'error': "The 'url' field is required."}, status=400)

    try:
        batch = download_photos(url, limit)
    except DownloadError as exc:
        status = 422 if exc.details else 400
        return JsonResponse({'error': str(exc), 'details': exc.details}, status=status)
    except requests.RequestException as exc:
        return JsonResponse({'error': f'Could not reach the link: {exc}'}, status=502)

    photos = batch.photos.defer('data')
    return JsonResponse({
        'batch_id': batch.pk,
        'url': batch.source_url,
        'provider': batch.provider,
        'count': len(photos),
        'photos': [
            {
                'id': photo.pk,
                'filename': photo.filename,
                'content_type': photo.content_type,
                'size': photo.size,
                'url': request.build_absolute_uri(photo.get_absolute_url()),
            }
            for photo in photos
        ],
        'errors': batch.errors,
        'preview_url': request.build_absolute_uri(f"{reverse('gallery')}#batch-{batch.pk}"),
    })


def _gallery_photos():
    return Photo.objects.defer('data').select_related('matched_face').defer('matched_face__embedding')


@require_GET
def gallery(request):
    """Preview page listing every downloaded photo, grouped by share link.

    With ?view=matches it lists only the photos whose face matches the profile, best first.
    """
    match_percent = settings.FACE_MATCH_PERCENT
    matches = _gallery_photos().filter(match_percent__gte=match_percent)
    match_job = MatchJob.objects.first()
    if match_job:
        match_job = matching.refresh_if_stale(match_job)
    context = {
        'totals': Photo.objects.aggregate(count=Count('id'), size=Sum('size')),
        'match_count': matches.count(),
        'match_percent': match_percent,
        'match_job': match_job,
        'unscanned_count': Photo.objects.filter(faces_scanned=False).count(),
        'view': 'matches' if request.GET.get('view') == 'matches' else 'all',
    }
    if context['view'] == 'matches':
        context['page'] = Paginator(matches.order_by('-match_percent', 'id'), 60).get_page(request.GET.get('page'))
    else:
        batches = (
            DownloadBatch.objects
            .annotate(photo_count=Count('photos'), total_size=Sum('photos__size'))
            .prefetch_related(Prefetch('photos', queryset=_gallery_photos()))
        )
        context['page'] = Paginator(batches, 10).get_page(request.GET.get('page'))
    return render(request, 'photos/gallery.html', context)


@require_POST
def match_sync(request):
    """Start a face-match sync (the gallery's Sync button). Answers JSON when asked for it."""
    wants_json = 'application/json' in request.headers.get('Accept', '')
    try:
        job = matching.start_sync()
    except matching.MatchError as exc:
        if wants_json:
            return JsonResponse({'error': str(exc)}, status=400)
        messages.error(request, str(exc))
        return redirect('gallery')
    if wants_json:
        return JsonResponse(job.as_dict(), status=202)
    return redirect('gallery')


@require_GET
def match_job_status(request, pk):
    """JSON progress of a face-match sync, polled by the gallery."""
    job = matching.refresh_if_stale(get_object_or_404(MatchJob, pk=pk))
    return JsonResponse(job.as_dict())


@require_http_methods(['GET', 'POST'])
def add_link(request):
    """Form to submit a shared link; starts a background download and shows its progress."""
    source = request.POST if request.method == 'POST' else request.GET
    context = {
        'url': source.get('url', '').strip(),
        'limit': source.get('limit', '').strip(),
        'max_photos': MAX_PHOTOS,
        'max_photos_display': f'{MAX_PHOTOS:,}',
    }
    if request.method == 'POST':
        try:
            job = jobs.start_job(context['url'], context['limit'])
        except DownloadError as exc:
            context.update(error=str(exc), details=exc.details)
        else:
            return redirect('download-job', pk=job.pk)
    return render(request, 'photos/add_link.html', context)


@require_GET
def download_job(request, pk):
    """Progress page for a background download; moves on to the gallery once it is done."""
    job = jobs.refresh_if_stale(get_object_or_404(DownloadJob, pk=pk))
    if job.status == DownloadJob.DONE and job.batch_id:
        messages.success(
            request, f'Downloaded {job.saved} photo{"s" if job.saved != 1 else ""} from {job.source_url}'
        )
        return redirect(f"{reverse('gallery')}#batch-{job.batch_id}")
    retry_url = f"{reverse('add-link')}?{urlencode({'url': job.source_url, 'limit': job.limit or ''})}"
    return render(request, 'photos/download_job.html', {'job': job, 'retry_url': retry_url})


@require_GET
def download_job_status(request, pk):
    """JSON progress of a background download, polled by the progress page."""
    job = jobs.refresh_if_stale(get_object_or_404(DownloadJob, pk=pk))
    return JsonResponse(job.as_dict())


@require_POST
def delete_all(request):
    """Delete every downloaded photo along with the batches they came from."""
    count = Photo.objects.count()
    DownloadBatch.objects.all().delete()
    messages.success(request, f'Deleted {count} photo{"s" if count != 1 else ""}.')
    return redirect('gallery')


@require_GET
def photo_image(request, pk):
    """Serve a stored photo straight from the database."""
    photo = get_object_or_404(Photo, pk=pk)
    response = HttpResponse(bytes(photo.data), content_type=photo.content_type)
    disposition = 'attachment' if request.GET.get('download') else 'inline'
    response['Content-Disposition'] = f'{disposition}; filename="{photo.filename}"'
    # Stored photos never change, so browsers can cache them for a long time.
    response['Cache-Control'] = 'private, max-age=31536000, immutable'
    return response


@require_http_methods(['GET', 'POST'])
def profile_page(request):
    """Edit the app profile: display name and profile photo."""
    profile = Profile.load()
    errors = {}
    if request.method == 'POST':
        if request.POST.get('action') == 'remove-photo':
            profile.photo, profile.photo_content_type = None, ''
            matching.update_profile_face(profile)
            profile.save()
            messages.success(request, 'Profile photo removed.')
            return redirect('profile')

        name = request.POST.get('display_name', '').strip()
        if len(name) > 80:
            errors['display_name'] = 'The name can be at most 80 characters.'
        upload = request.FILES.get('photo')
        if upload:
            try:
                data, content_type = read_profile_photo(upload)
            except InvalidPhoto as exc:
                errors['photo'] = str(exc)
        if upload and not errors:
            old_photo = (profile.photo, profile.photo_content_type, profile.face_embedding)
            profile.photo, profile.photo_content_type = data, content_type
            try:
                matching.update_profile_face(profile)
            except faces.ProfileFaceError as exc:
                errors['photo'] = str(exc)
                profile.photo, profile.photo_content_type, profile.face_embedding = old_photo
        if not errors:
            profile.display_name = name
            profile.save()
            messages.success(request, 'Profile saved.')
            if upload:
                messages.info(request, 'Press "Sync face match" in the gallery to find this face in your photos.')
            return redirect('profile')
        profile.display_name = name  # keep what was typed
    return render(request, 'photos/profile.html', {'profile': profile, 'errors': errors})


@require_GET
def profile_photo(request):
    """Serve the profile photo from the database."""
    profile = Profile.load()
    if not profile.has_photo:
        return HttpResponse(status=404)
    response = HttpResponse(bytes(profile.photo), content_type=profile.photo_content_type)
    # Uploads are user-supplied: never let the browser treat them as anything but an image.
    response['X-Content-Type-Options'] = 'nosniff'
    response['Cache-Control'] = 'private, max-age=31536000, immutable' if request.GET.get('v') else 'no-cache'
    return response
