from django.shortcuts import render


def Landing(request):
    return render(request, "Guest/landing.html")

def Login(request):
    return render(request, "Guest/login.html")
