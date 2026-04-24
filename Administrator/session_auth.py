from functools import wraps

from django.shortcuts import redirect

from .models import Administrators


APP_ADMIN_SESSION_KEY = "app_admin_id"


def get_current_admin(request):
    admin_id = request.session.get(APP_ADMIN_SESSION_KEY)
    if not admin_id:
        return None

    try:
        return Administrators.objects.get(pk=admin_id)
    except Administrators.DoesNotExist:
        request.session.pop(APP_ADMIN_SESSION_KEY, None)
        request.session.modified = True
        return None


def login_app_admin(request, admin_user):
    request.session.flush()
    request.session[APP_ADMIN_SESSION_KEY] = admin_user.pk
    request.session.modified = True


def clear_app_admin_session(request):
    if APP_ADMIN_SESSION_KEY in request.session:
        request.session.pop(APP_ADMIN_SESSION_KEY, None)
        request.session.modified = True


def logout_app_admin(request):
    request.session.flush()


def admin_login_required(view_func):
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        if get_current_admin(request) is None:
            return redirect(f"/login/?next={request.get_full_path()}")
        return view_func(request, *args, **kwargs)

    return wrapped
