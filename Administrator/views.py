from urllib.parse import urlencode

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .session_auth import admin_login_required, get_current_admin, logout_app_admin
from User.models import Users
from User.notifications import send_registration_status_email


def Login(request):
    if get_current_admin(request) is not None:
        return redirect("Administrator:dashboard")

    next_url = request.GET.get("next") or request.POST.get("next") or ""
    target_next = next_url or "/admin-panel/"
    login_url = f"/login/?{urlencode({'next': target_next})}"
    return redirect(login_url)


@admin_login_required
def Dashboard(request):
    users = Users.objects.all().order_by("status", "-created_at", "name")
    context = {
        "users": users,
        "pending_count": users.filter(status=Users.STATUS_PENDING).count(),
        "accepted_count": users.filter(status=Users.STATUS_ACCEPTED).count(),
        "rejected_count": users.filter(status=Users.STATUS_REJECTED).count(),
    }
    return render(request, "Administrator/dashboard.html", context)


@require_POST
@admin_login_required
def AcceptUser(request, user_id):
    return _update_user_status(request, user_id, Users.STATUS_ACCEPTED)


@require_POST
@admin_login_required
def RejectUser(request, user_id):
    return _update_user_status(request, user_id, Users.STATUS_REJECTED)


@admin_login_required
def Logout(request):
    logout_app_admin(request)
    messages.info(request, "Administrator logout successful.")
    return redirect("Guest:login")


def _update_user_status(request, user_id, new_status):
    user = get_object_or_404(Users, pk=user_id)
    previous_status = user.status

    if previous_status == new_status:
        messages.info(request, f"{user.name} is already marked as {user.get_status_display().lower()}.")
        return redirect("Administrator:dashboard")

    user.status = new_status
    user.save(update_fields=["status", "updated_at"])

    email_sent = send_registration_status_email(user)
    status_label = user.get_status_display().lower()

    if email_sent:
        messages.success(request, f"{user.name} has been {status_label} and an email was sent.")
    else:
        messages.success(request, f"{user.name} has been {status_label}. Email could not be sent.")

    return redirect("Administrator:dashboard")
