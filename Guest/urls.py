from django.urls import path
from Guest import views

app_name = "Guest"

urlpatterns = [
    path('', views.Landing, name="landing"),
    path('login/', views.Login, name="login"),

  
]