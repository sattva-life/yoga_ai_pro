from django.contrib import admin

from .models import Users


@admin.register(Users)
class UsersAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "email", "status", "created_at")
    search_fields = ("name", "email")
    readonly_fields = ("created_at", "updated_at")
    list_filter = ("status", "created_at")
