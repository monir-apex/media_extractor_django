from django.urls import path

from . import views

urlpatterns = [
    path('api/photos/download/', views.download_from_link, name='download-from-link'),
    path('photos/', views.gallery, name='gallery'),
    path('api/photos/jobs/<uuid:pk>/', views.download_job_status, name='download-job-status'),
    path('photos/add/', views.add_link, name='add-link'),
    path('photos/jobs/<uuid:pk>/', views.download_job, name='download-job'),
    path('photos/match/sync/', views.match_sync, name='match-sync'),
    path('api/photos/match/jobs/<uuid:pk>/', views.match_job_status, name='match-job-status'),
    path('photos/delete-all/', views.delete_all, name='delete-all'),
    path('photos/<int:pk>/image/', views.photo_image, name='photo-image'),
    path('profile/', views.profile_page, name='profile'),
    path('profile/photo/', views.profile_photo, name='profile-photo'),
]
