from django.contrib import admin

from .models import DownloadBatch, DownloadJob, Photo, Profile


class PhotoInline(admin.TabularInline):
    model = Photo
    fields = ('filename', 'content_type', 'size', 'created_at')
    readonly_fields = fields
    extra = 0
    can_delete = False


@admin.register(DownloadBatch)
class DownloadBatchAdmin(admin.ModelAdmin):
    list_display = ('id', 'provider', 'source_url', 'created_at')
    list_filter = ('provider',)
    inlines = [PhotoInline]


@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ('id', 'filename', 'content_type', 'size', 'batch', 'created_at')
    list_filter = ('content_type',)
    exclude = ('data',)

    def get_queryset(self, request):
        return super().get_queryset(request).defer('data')


@admin.register(DownloadJob)
class DownloadJobAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'status', 'source_url', 'saved', 'percent')
    list_filter = ('status',)
    readonly_fields = [f.name for f in DownloadJob._meta.fields]


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('display_name', 'photo_content_type', 'updated_at')
    exclude = ('photo',)
