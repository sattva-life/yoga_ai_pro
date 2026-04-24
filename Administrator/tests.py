import shutil
import tempfile

from django.core import mail
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from Administrator.models import Administrators
from User.models import Users


class AdministratorApprovalTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.temp_media_root = tempfile.mkdtemp()
        cls.override = override_settings(MEDIA_ROOT=cls.temp_media_root)
        cls.override.enable()

    @classmethod
    def tearDownClass(cls):
        cls.override.disable()
        shutil.rmtree(cls.temp_media_root, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        self.admin_user = Administrators(name="SattvaLife Admin", email="admin@example.com")
        self.admin_user.set_password("superpass123")
        self.admin_user.save()

    def make_photo(self):
        return SimpleUploadedFile(
            "photo.gif",
            (
                b"GIF89a\x01\x00\x01\x00\x80\x00\x00"
                b"\x00\x00\x00\xff\xff\xff!\xf9\x04\x00"
                b"\x00\x00\x00\x00,\x00\x00\x00\x00\x01"
                b"\x00\x01\x00\x00\x02\x02D\x01\x00;"
            ),
            content_type="image/gif",
        )

    def create_registered_user(self, status=Users.STATUS_PENDING):
        user = Users(
            name="Nila",
            email=f"nila{status}@example.com",
            address="Ernakulam",
            photo=self.make_photo(),
            status=status,
        )
        user.set_password("securepass123")
        user.save()
        return user

    def test_dashboard_requires_administrator_login(self):
        response = self.client.get("/admin-panel/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/?next=/admin-panel/", response.headers["Location"])

    def test_administrator_can_login_with_superuser_credentials_on_shared_login_page(self):
        response = self.client.post(
            "/login/",
            {"identifier": "admin@example.com", "password": "superpass123", "next": "/admin-panel/"},
            follow=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/admin-panel/")
        self.assertEqual(self.client.session.get("app_admin_id"), self.admin_user.pk)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_admin_login_route_redirects_to_shared_login_page(self):
        response = self.client.get("/admin-panel/login/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/?next=%2Fadmin-panel%2F", response.headers["Location"])

    def test_administrator_can_accept_pending_user_and_send_email(self):
        session = self.client.session
        session["app_admin_id"] = self.admin_user.pk
        session.save()
        user = self.create_registered_user()

        response = self.client.post(f"/admin-panel/users/{user.id}/accept/", follow=False)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/admin-panel/")
        user.refresh_from_db()
        self.assertEqual(user.status, Users.STATUS_ACCEPTED)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("approved", mail.outbox[0].subject.lower())
        self.assertIn("approved", mail.outbox[0].body.lower())

    def test_administrator_can_reject_pending_user_and_send_email(self):
        session = self.client.session
        session["app_admin_id"] = self.admin_user.pk
        session.save()
        user = self.create_registered_user()

        response = self.client.post(f"/admin-panel/users/{user.id}/reject/", follow=False)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/admin-panel/")
        user.refresh_from_db()
        self.assertEqual(user.status, Users.STATUS_REJECTED)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("rejected", mail.outbox[0].subject.lower())
        self.assertIn("rejected", mail.outbox[0].body.lower())
