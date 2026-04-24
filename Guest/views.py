from django.contrib import messages
from django.shortcuts import redirect, render

from Administrator.models import Administrators
from Administrator.session_auth import get_current_admin, login_app_admin
from User.models import Users
from User.notifications import send_registration_received_email
from User.session_auth import (
    get_current_user,
    get_safe_redirect,
    login_app_user,
    logout_app_user,
)


def Landing(request):
    return render(request, "Guest/landing.html", {"current_user": get_current_user(request)})


def Login(request):
    current_admin = get_current_admin(request)
    if current_admin is not None:
        return redirect("Administrator:dashboard")

    current_user = get_current_user(request)
    if current_user is not None:
        return redirect("User:home_page")

    next_url = request.GET.get("next") or request.POST.get("next") or ""
    admin_requested = next_url.startswith("/admin-panel/")

    if request.method == "POST":
        identifier = str(request.POST.get("identifier") or request.POST.get("email", "")).strip()
        normalized_identifier = identifier.lower()
        password = request.POST.get("password", "")
        form_context = {"identifier_value": identifier, "next_url": next_url}

        admin_user = Administrators.objects.filter(email__iexact=normalized_identifier).first()
        if admin_user is not None and admin_user.check_password(password):
            login_app_admin(request, admin_user)
            messages.success(request, f"Welcome back, {admin_user.name}.")
            return redirect(get_safe_redirect(request, fallback="/admin-panel/"))

        user = Users.objects.filter(email__iexact=normalized_identifier).first()
        if user is not None and user.check_password(password):
            if user.is_pending:
                messages.error(request, "Your account is under verification. Please wait for administrator approval.")
                return render(request, "Guest/login.html", form_context)

            if user.is_rejected:
                messages.error(request, "Your account has been rejected. Please contact the administrator.")
                return render(request, "Guest/login.html", form_context)

            login_app_user(request, user)
            messages.success(request, f"Welcome back, {user.name}.")
            if admin_requested:
                return redirect("User:home_page")
            return redirect(get_safe_redirect(request, fallback="/user/"))

        messages.error(request, "Invalid email or password.")
        return render(request, "Guest/login.html", form_context)

    return render(request, "Guest/login.html", {"next_url": next_url})


def Signup(request):
    current_admin = get_current_admin(request)
    if current_admin is not None:
        return redirect("Administrator:dashboard")

    current_user = get_current_user(request)
    if current_user is not None:
        return redirect("User:home_page")

    if request.method == "POST":
        name = str(request.POST.get("name", "")).strip()
        email = str(request.POST.get("email", "")).strip().lower()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")
        address = str(request.POST.get("address", "")).strip()
        photo = request.FILES.get("photo")

        form_data = {
            "name_value": name,
            "email_value": email,
            "address_value": address,
        }

        if not all([name, email, password, confirm_password, address, photo]):
            messages.error(request, "Please complete all signup fields, including your photo.")
            return render(request, "Guest/signup.html", form_data)

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, "Guest/signup.html", form_data)

        if len(password) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            return render(request, "Guest/signup.html", form_data)

        if Users.objects.filter(email__iexact=email).exists():
            messages.error(request, "An account with this email already exists.")
            return render(request, "Guest/signup.html", form_data)

        user = Users(
            name=name,
            email=email,
            address=address,
            photo=photo,
            status=Users.STATUS_PENDING,
        )
        user.set_password(password)
        user.save()

        email_sent = send_registration_received_email(user)
        if email_sent:
            messages.success(
                request,
                "Registration completed. Your account is under verification. Please wait for approval. A confirmation email has been sent to you.",
            )
        else:
            messages.success(
                request,
                "Registration completed. Your account is under verification. Please wait for approval.",
            )
        return redirect("Guest:login")

    return render(request, "Guest/signup.html")


def Logout(request):
    logout_app_user(request)
    messages.info(request, "You have been logged out.")
    return redirect("Guest:landing")
