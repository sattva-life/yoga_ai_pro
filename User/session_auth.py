from functools import wraps

from django.http import JsonResponse
from django.shortcuts import redirect
from django.utils.http import url_has_allowed_host_and_scheme

from .models import Users


APP_USER_SESSION_KEY = "app_user_id"


def get_current_user(request):
    user_id = request.session.get(APP_USER_SESSION_KEY)
    if not user_id:
        return None

    try:
        user = Users.objects.get(pk=user_id)
    except Users.DoesNotExist:
        request.session.pop(APP_USER_SESSION_KEY, None)
        request.session.modified = True
        return None

    if not user.is_accepted:
        request.session.pop(APP_USER_SESSION_KEY, None)
        request.session.modified = True
        return None

    return user


def login_app_user(request, user):
    request.session.flush()
    request.session[APP_USER_SESSION_KEY] = user.pk
    request.session.modified = True


def clear_app_user_session(request):
    if APP_USER_SESSION_KEY in request.session:
        request.session.pop(APP_USER_SESSION_KEY, None)
        request.session.modified = True


def logout_app_user(request):
    request.session.flush()


def get_safe_redirect(request, fallback):
    next_url = request.GET.get("next") or request.POST.get("next")
    if next_url and url_has_allowed_host_and_scheme(
        next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ):
        return next_url
    return fallback


def app_login_required(view_func):
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        if get_current_user(request) is None:
            return redirect(f"/login/?next={request.get_full_path()}")
        return view_func(request, *args, **kwargs)

    return wrapped


def app_api_login_required(view_func):
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        if get_current_user(request) is None:
            return JsonResponse(
                {
                    "success": False,
                    "error": "Authentication required",
                    "login_url": "/login/",
                },
                status=401,
            )
        return view_func(request, *args, **kwargs)

    return wrapped
