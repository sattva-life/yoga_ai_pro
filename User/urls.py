from django.urls import path
from . import views

app_name = "User"

urlpatterns = [
    path("", views.HomePage, name="home_page"),
    path("camera/", views.camera_page, name="camera_page"),
    path("predict-yoga-pose/", views.predict_yoga_pose, name="predict_yoga_pose"),

    path("down-dog-live/", views.down_dog_live_page, name="down_dog_live_page"),
    path("down-dog-live-api/", views.down_dog_live_api, name="down_dog_live_api"),

    path("goddess-live/", views.goddess_live_page, name="goddess_live_page"),
    path("predict-goddess-pose/", views.predict_goddess_pose, name="predict_goddess_pose"),

    path("plank-live/", views.plank_live_page, name="plank_live_page"),
    path("predict-plank-pose/", views.predict_plank_pose, name="predict_plank_pose"),
    
    path("warrior-live/", views.warrior_live_page, name="warrior_live_page"),
    path("predict-warrior-pose/", views.predict_warrior_pose, name="predict_warrior_pose"),
]