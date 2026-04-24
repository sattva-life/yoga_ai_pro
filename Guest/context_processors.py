from Administrator.session_auth import get_current_admin
from User.session_auth import get_current_user


def session_accounts(request):
    return {
        "current_user": get_current_user(request),
        "current_admin": get_current_admin(request),
    }
