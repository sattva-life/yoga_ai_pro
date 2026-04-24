from django.contrib import admin

from .models import Administrators


@admin.register(Administrators)
class AdministratorsAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "email", "created_at")
    search_fields = ("name", "email")
    readonly_fields = ("created_at", "updated_at")
