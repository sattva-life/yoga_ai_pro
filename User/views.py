from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from .utils.tree_utility import process_yoga_pose_request
from .utils.down_dog_utility import process_down_dog_request


def HomePage(request):
    return render(request, "User/home_page.html")


def camera_page(request):
    return render(request, "User/camera.html")


def down_dog_live_page(request):
    return render(request, "User/downdog_camera.html")


def api_error(message, status=400):
    return JsonResponse({"success": False, "error": str(message)}, status=status)


@csrf_exempt
def predict_yoga_pose(request):
    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)

    if "image" not in request.FILES:
        return api_error("No image uploaded", status=400)

    return process_yoga_pose_request(request)


@csrf_exempt
def down_dog_live_api(request):
    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)

    if "image" not in request.FILES:
        return api_error("No image uploaded", status=400)

    return process_down_dog_request(request)