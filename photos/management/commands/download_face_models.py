from django.core.management.base import BaseCommand, CommandError

from photos import faces


class Command(BaseCommand):
    help = "Download the face detection and recognition models used by face matching."

    def handle(self, *args, **options):
        try:
            faces.ensure_models()
        except faces.FaceError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"Face models are ready in {faces.models_dir()}"))
