# User/urls.py

from django.urls import path
from User import views

app_name = "User"

urlpatterns = [
    path("", views.HomePage, name="home_page"),
    path("camera/", views.camera_page, name="camera_page"),
    path("predict-yoga-pose/", views.predict_yoga_pose, name="predict_yoga_pose"),
]