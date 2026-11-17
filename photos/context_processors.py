from .models import Profile


def profile(request):
    """Make the app profile available to every page (for the avatar in the header)."""
    return {'profile': Profile.objects.defer('photo').get_or_create(pk=1)[0]}
