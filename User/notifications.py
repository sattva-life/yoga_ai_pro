from django.conf import settings
from django.core.mail import send_mail

from .models import Users


def email_notifications_enabled():
    return bool(settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD)


def send_registration_received_email(user):
    subject = "SattvaLife registration received"
    message = (
        f"Hello {user.name},\n\n"
        "Your registration has been received successfully.\n"
        "Your account is currently under verification. Please wait for administrator approval.\n\n"
        "We will send you another email once your account is accepted or rejected.\n\n"
        "Thanks,\n"
        "SattvaLife Yoga"
    )
    return _send_user_email(user.email, subject, message)


def send_registration_status_email(user):
    if user.status == Users.STATUS_ACCEPTED:
        subject = "Your SattvaLife account has been approved"
        message = (
            f"Hello {user.name},\n\n"
            "Your account has been approved.\n"
            "You can now log in and start using SattvaLife Yoga.\n\n"
            "Thanks,\n"
            "SattvaLife Yoga"
        )
    elif user.status == Users.STATUS_REJECTED:
        subject = "Your SattvaLife account has been rejected"
        message = (
            f"Hello {user.name},\n\n"
            "Your account request has been rejected by the administrator.\n"
            "If you believe this was a mistake, please contact support.\n\n"
            "Thanks,\n"
            "SattvaLife Yoga"
        )
    else:
        return False

    return _send_user_email(user.email, subject, message)


def _send_user_email(recipient, subject, message):
    if not email_notifications_enabled():
        return False

    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient],
            fail_silently=False,
        )
        return True
    except Exception:
        return False
