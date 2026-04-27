from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
from django.core.mail import EmailMessage

from .session_auth import app_api_login_required, app_login_required, get_current_user
from .utils.tree_utility import process_yoga_pose_request
from .utils.down_dog_utility import process_down_dog_request
from .utils.goddess_utility import process_goddess_pose_request # New Import

from .utils.warrior_utility import process_warrior_pose_request # New Import

@app_login_required
def HomePage(request):
    return render(request, "User/home_page.html", {"current_user": get_current_user(request)})



@app_login_required
def camera_page(request):
    return render(request, "User/tree_camera.html", {"current_user": get_current_user(request)})


@app_login_required
def warrior_live_page(request):
    return render(request, "User/warrior_camera.html", {"current_user": get_current_user(request)})

@app_api_login_required
@csrf_exempt
def predict_warrior_pose(request):
    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)
    if "image" not in request.FILES:
        return api_error("No image uploaded", status=400)
    return process_warrior_pose_request(request)




@app_login_required
def down_dog_live_page(request):
    return render(request, "User/downdog_camera.html", {"current_user": get_current_user(request)})

@app_login_required
def goddess_live_page(request):
    return render(request, "User/goddess_camera.html", {"current_user": get_current_user(request)}) # New Page View

def api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)


@app_api_login_required
@csrf_exempt
def email_pose_report(request):
    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)

    report_file = request.FILES.get("report")
    if report_file is None:
        return api_error("No report attached", status=400)

    current_user = get_current_user(request)
    if current_user is None:
        return api_error("Authentication required", status=401)

    pose_name = str(request.POST.get("pose") or "Yoga Pose").strip()[:80]
    filename = report_file.name or f"Sattvalife_{pose_name.replace(' ', '_')}_Report.pdf"
    content_type = report_file.content_type or "application/pdf"

    if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        return api_error("Only PDF reports can be emailed", status=400)

    subject = f"SattvaLife {pose_name} report"
    message = (
        f"Hello {current_user.name},\n\n"
        f"Your {pose_name} practice report is attached to this email.\n\n"
        "Keep practicing steadily.\n\n"
        "Thanks,\n"
        "SattvaLife Yoga"
    )

    try:
        email = EmailMessage(
            subject=subject,
            body=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[current_user.email],
        )
        email.attach(filename, report_file.read(), content_type)
        email.send(fail_silently=False)
    except Exception:
        return api_error("Could not send report email. Please check email settings.", status=500)

    return JsonResponse({"success": True, "message": "Report emailed successfully."})


@app_api_login_required
@csrf_exempt
def predict_yoga_pose(request):
    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)

    if "image" not in request.FILES:
        return api_error("No image uploaded", status=400)

    return process_yoga_pose_request(request)


@app_api_login_required
@csrf_exempt
def down_dog_live_api(request):
    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)

    if "image" not in request.FILES:
        return api_error("No image uploaded", status=400)

    return process_down_dog_request(request)


@app_api_login_required
@csrf_exempt
def predict_goddess_pose(request): # New API Endpoint
    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)
    if "image" not in request.FILES:
        return api_error("No image uploaded", status=400)
    return process_goddess_pose_request(request)


