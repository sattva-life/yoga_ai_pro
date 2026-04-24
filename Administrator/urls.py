from django.urls import path
from . import views

app_name = "Administrator"

urlpatterns = [
    path("", views.Dashboard, name="dashboard"),
    path("login/", views.Login, name="login"),
    path("logout/", views.Logout, name="logout"),
    path("users/<int:user_id>/accept/", views.AcceptUser, name="accept_user"),
    path("users/<int:user_id>/reject/", views.RejectUser, name="reject_user"),
]
