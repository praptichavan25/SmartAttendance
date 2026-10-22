"""
test_guest_security.py
----------------------
Comprehensive test suite verifying:
1. Role-based restrictions on guest registration (student/teacher 403, security/admin 200).
2. Single-face validation for guest registration (rejects invalid/empty image).
3. Successful guest registration with face template extraction and DB persistence.
4. Active guest recognized as APPROVED_GUEST (orange box) with correct name and ID.
5. Different unknown face recognized as UNKNOWN_PERSON (red box).
6. Revocation of guest immediately causes next recognition to return UNKNOWN_PERSON (red box).
7. Expired guest passes are not matched as approved guests.
8. Guest recognition NEVER creates student attendance records.
9. Security events recording and resolution audit trail.
10. Sensitive biometrics (face_embedding) are never exposed via GET /api/guests.
"""

import base64
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
import unittest

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import get_db, init_db
from app.auth import create_access_token


class TestGuestSecurity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

        # Ensure a clean slate for synthetic test guests
        with get_db() as conn:
            conn.execute("DELETE FROM guests WHERE full_name IN ('Dr. Rajesh Gupta', 'Expired Guest')")

        cls.student_photo = Path(__file__).resolve().parent.parent.parent / "ml/dataset/test/registered/Prapti/test_photo.jpg"
        cls.guest_photo = Path(__file__).resolve().parent.parent.parent / "ml/dataset/test/unknown/Unknown2/test_photo2.jpg"
        cls.other_unknown_photo = Path(__file__).resolve().parent.parent.parent / "ml/dataset/test/unknown/Unknown3/test_photo3.jpg"

        assert cls.student_photo.exists(), f"Missing student photo at {cls.student_photo}"
        assert cls.guest_photo.exists(), f"Missing guest photo at {cls.guest_photo}"
        assert cls.other_unknown_photo.exists(), f"Missing other unknown photo at {cls.other_unknown_photo}"

    @classmethod
    def tearDownClass(cls):
        with get_db() as conn:
            conn.execute("DELETE FROM guests WHERE full_name IN ('Dr. Rajesh Gupta', 'Expired Guest')")

    def get_token_headers(self, username: str, role: str) -> dict:
        with get_db() as conn:
            user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            assert user, f"User {username} must exist"
            user_id = user["id"]
        token = create_access_token({"sub": str(user_id), "role": role})
        return {"Authorization": f"Bearer {token}"}

    def test_01_rbac_guest_registration_forbidden_for_students_and_teachers(self):
        student_headers = self.get_token_headers("prapti", "student")
        teacher_headers = self.get_token_headers("teacher", "teacher")

        payload = {
            "full_name": "Test Visitor",
            "purpose": "Campus Tour",
            "host_to_meet": "Prof. Sharma",
            "image_base64": "invalid",
        }

        res_student = self.client.post("/api/guests/register", headers=student_headers, json=payload)
        self.assertEqual(res_student.status_code, 403)

        res_teacher = self.client.post("/api/guests/register", headers=teacher_headers, json=payload)
        self.assertEqual(res_teacher.status_code, 403)

    def test_02_register_guest_rejects_invalid_images(self):
        security_headers = self.get_token_headers("security", "security")

        # Invalid base64
        res1 = self.client.post(
            "/api/guests/register",
            headers=security_headers,
            json={
                "full_name": "Invalid Visitor",
                "purpose": "Meeting",
                "host_to_meet": "Dean",
                "image_base64": "not_a_valid_base64_string!@#",
            },
        )
        self.assertEqual(res1.status_code, 400)

        # Non-face image (e.g. blank 1x1 image or text)
        blank_b64 = base64.b64encode(b"not_an_image").decode("utf-8")
        res2 = self.client.post(
            "/api/guests/register",
            headers=security_headers,
            json={
                "full_name": "Invalid Visitor",
                "purpose": "Meeting",
                "host_to_meet": "Dean",
                "image_base64": blank_b64,
            },
        )
        self.assertEqual(res2.status_code, 400)

    def test_03_security_registers_guest_successfully(self):
        security_headers = self.get_token_headers("security", "security")

        with open(self.guest_photo, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "full_name": "Dr. Rajesh Gupta",
            "purpose": "Guest Lecture on AI",
            "host_to_meet": "Prof. Deshmukh",
            "phone": "+91 9876543210",
            "valid_from": datetime.now().strftime("%Y-%m-%dT00:00:00"),
            "valid_until": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59"),
            "image_base64": img_b64,
        }

        res = self.client.post("/api/guests/register", headers=security_headers, json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        TestGuestSecurity.registered_guest_id = data["guest_id"]
        self.assertTrue(data["guest_id"].startswith("GST-"))
        self.assertEqual(data["full_name"], "Dr. Rajesh Gupta")
        self.assertEqual(data["status"], "approved")
        self.assertEqual(data["approved_by"], "security")
        self.assertTrue(data["face_snapshot"].startswith("data:image/jpeg;base64,"))

        # Check in DB
        with get_db() as conn:
            guest_row = conn.execute("SELECT * FROM guests WHERE guest_id = ?", (data["guest_id"],)).fetchone()
            self.assertIsNotNone(guest_row)
            self.assertEqual(guest_row["full_name"], "Dr. Rajesh Gupta")
            self.assertEqual(len(guest_row["face_embedding"]), 512)  # 128 floats * 4 bytes = 512 bytes

    def test_04_guest_recognized_as_approved_guest_orange_box(self):
        security_headers = self.get_token_headers("security", "security")

        # Recognize with guest_photo (now registered as Dr. Rajesh Gupta)
        with open(self.guest_photo, "rb") as f:
            res = self.client.post("/api/recognize", headers=security_headers, files={"file": ("guest.jpg", f, "image/jpeg")})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["faces_detected"], 1)

        result = data["results"][0]
        self.assertEqual(result["identity_category"], "APPROVED_GUEST")
        self.assertEqual(result["authorization_status"], "AUTHORIZED")
        self.assertEqual(result["box_color"], "orange")
        self.assertEqual(result["name"], "Dr. Rajesh Gupta")
        self.assertTrue(result["guest_id"].startswith("GST-"))

    def test_05_registered_student_recognized_as_student_green_box(self):
        teacher_headers = self.get_token_headers("teacher", "teacher")

        # Recognize with Prapti's photo
        with open(self.student_photo, "rb") as f:
            res = self.client.post("/api/recognize", headers=teacher_headers, files={"file": ("prapti.jpg", f, "image/jpeg")})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["faces_detected"], 1)

        result = data["results"][0]
        self.assertEqual(result["identity_category"], "REGISTERED_STUDENT")
        self.assertEqual(result["authorization_status"], "AUTHORIZED")
        self.assertEqual(result["box_color"], "green")
        self.assertEqual(result["name"], "Prapti")

    def test_06_unregistered_person_recognized_as_unknown_red_box(self):
        security_headers = self.get_token_headers("security", "security")

        # Recognize with other_unknown_photo (unregistered)
        with open(self.other_unknown_photo, "rb") as f:
            res = self.client.post("/api/recognize", headers=security_headers, files={"file": ("other.jpg", f, "image/jpeg")})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["faces_detected"], 1)

        result = data["results"][0]
        self.assertEqual(result["identity_category"], "UNKNOWN_PERSON")
        self.assertEqual(result["authorization_status"], "UNAUTHORIZED")
        self.assertEqual(result["box_color"], "red")
        self.assertEqual(result["name"], "UNKNOWN PERSON")

    def test_07_guest_revocation_immediately_reverts_to_unknown_red_box(self):
        security_headers = self.get_token_headers("security", "security")
        guest_id = getattr(TestGuestSecurity, "registered_guest_id", None)
        self.assertIsNotNone(guest_id)

        # Revoke pass
        res_revoke = self.client.post(
            f"/api/guests/{guest_id}/revoke",
            headers=security_headers,
            json={"reason": "Completed lecture and left campus"},
        )
        self.assertEqual(res_revoke.status_code, 200)
        self.assertEqual(res_revoke.json()["status"], "revoked")

        # Immediate re-recognition of same person
        with open(self.guest_photo, "rb") as f:
            res_rec = self.client.post("/api/recognize", headers=security_headers, files={"file": ("guest.jpg", f, "image/jpeg")})

        self.assertEqual(res_rec.status_code, 200)
        result = res_rec.json()["results"][0]
        self.assertEqual(result["identity_category"], "UNKNOWN_PERSON")
        self.assertEqual(result["authorization_status"], "UNAUTHORIZED")
        self.assertEqual(result["box_color"], "red")
        self.assertEqual(result["name"], "UNKNOWN PERSON")

    def test_08_expired_guest_is_not_authorized(self):
        security_headers = self.get_token_headers("security", "security")

        with open(self.other_unknown_photo, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Register guest with expired time window (yesterday)
        yesterday_start = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00")
        yesterday_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT23:59:59")

        res = self.client.post(
            "/api/guests/register",
            headers=security_headers,
            json={
                "full_name": "Expired Guest",
                "purpose": "Past Event",
                "host_to_meet": "Staff",
                "valid_from": yesterday_start,
                "valid_until": yesterday_end,
                "image_base64": img_b64,
            },
        )
        self.assertEqual(res.status_code, 200)

        # Recognition should reject as UNKNOWN_PERSON because validity has expired
        with open(self.other_unknown_photo, "rb") as f:
            res_rec = self.client.post("/api/recognize", headers=security_headers, files={"file": ("other.jpg", f, "image/jpeg")})

        self.assertEqual(res_rec.status_code, 200)
        result = res_rec.json()["results"][0]
        self.assertEqual(result["identity_category"], "UNKNOWN_PERSON")
        self.assertEqual(result["authorization_status"], "UNAUTHORIZED")
        self.assertEqual(result["box_color"], "red")

    def test_09_guest_list_does_not_expose_raw_embeddings(self):
        teacher_headers = self.get_token_headers("teacher", "teacher")
        res = self.client.get("/api/guests", headers=teacher_headers)
        self.assertEqual(res.status_code, 200)
        guests = res.json()
        self.assertGreater(len(guests), 0)
        for g in guests:
            self.assertNotIn("face_embedding", g)
            self.assertIn("guest_id", g)
            self.assertIn("full_name", g)
            self.assertIn("status", g)

    def test_10_security_events_retrieval_and_resolution(self):
        security_headers = self.get_token_headers("security", "security")

        res_events = self.client.get("/api/security/events", headers=security_headers)
        self.assertEqual(res_events.status_code, 200)
        events = res_events.json()
        self.assertGreater(len(events), 0)

        unresolved_event = next((e for e in events if e["resolution_status"] == "unresolved"), None)
        if unresolved_event:
            evt_id = unresolved_event["event_id"]
            res_resolve = self.client.post(
                f"/api/security/events/{evt_id}/resolve",
                headers=security_headers,
                json={"resolution_status": "dismissed", "resolution_notes": "Security verified false alarm"},
            )
            self.assertEqual(res_resolve.status_code, 200)
            self.assertEqual(res_resolve.json()["resolution_status"], "dismissed")

    def test_11_guest_recognition_never_creates_student_attendance(self):
        security_headers = self.get_token_headers("security", "security")

        with get_db() as conn:
            att_before = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]

        # Call recognize endpoint with guest photo
        with open(self.guest_photo, "rb") as f:
            res = self.client.post("/api/recognize", headers=security_headers, files={"file": ("guest.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)

        with get_db() as conn:
            att_after = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]

        self.assertEqual(att_before, att_after, "Guest recognition must NEVER mark attendance in the students table")


if __name__ == "__main__":
    unittest.main()
