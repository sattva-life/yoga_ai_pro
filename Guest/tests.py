import shutil
import tempfile

from django.contrib.messages import get_messages
from django.core import mail
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from User.models import Users


class GuestAuthTests(TestCase):
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

    def test_signup_creates_pending_user_and_sends_verification_email(self):
        response = self.client.post(
            "/signup/",
            {
                "name": "Asha",
                "email": "asha@example.com",
                "password": "securepass123",
                "confirm_password": "securepass123",
                "address": "Kochi, Kerala",
                "photo": self.make_photo(),
            },
            follow=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/login/")

        created_user = Users.objects.get(email="asha@example.com")
        self.assertEqual(created_user.name, "Asha")
        self.assertNotEqual(created_user.password, "securepass123")
        self.assertTrue(created_user.check_password("securepass123"))
        self.assertEqual(created_user.status, Users.STATUS_PENDING)
        self.assertIsNone(self.client.session.get("app_user_id"))
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("under verification", mail.outbox[0].body.lower())

    def test_login_uses_custom_users_table_for_accepted_users_only(self):
        user = Users(
            name="Dev",
            email="dev@example.com",
            address="Thrissur",
            photo=self.make_photo(),
            status=Users.STATUS_ACCEPTED,
        )
        user.set_password("securepass123")
        user.save()

        response = self.client.post(
            "/login/",
            {"identifier": "dev@example.com", "password": "securepass123"},
            follow=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/user/")
        self.assertEqual(self.client.session.get("app_user_id"), user.pk)

    def test_login_blocks_pending_users(self):
        user = Users(
            name="Maya",
            email="maya@example.com",
            address="Kochi",
            photo=self.make_photo(),
            status=Users.STATUS_PENDING,
        )
        user.set_password("securepass123")
        user.save()

        response = self.client.post(
            "/login/",
            {"identifier": "maya@example.com", "password": "securepass123"},
            follow=False,
        )

        messages = [message.message for message in get_messages(response.wsgi_request)]

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.client.session.get("app_user_id"))
        self.assertTrue(any("under verification" in message.lower() for message in messages))

    def test_login_blocks_rejected_users(self):
        user = Users(
            name="Tara",
            email="tara@example.com",
            address="Kozhikode",
            photo=self.make_photo(),
            status=Users.STATUS_REJECTED,
        )
        user.set_password("securepass123")
        user.save()

        response = self.client.post(
            "/login/",
            {"identifier": "tara@example.com", "password": "securepass123"},
            follow=False,
        )

        messages = [message.message for message in get_messages(response.wsgi_request)]

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.client.session.get("app_user_id"))
        self.assertTrue(any("rejected" in message.lower() for message in messages))

    def test_protected_user_page_redirects_to_login(self):
        response = self.client.get("/user/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/?next=/user/", response.headers["Location"])
