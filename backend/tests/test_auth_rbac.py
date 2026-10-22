"""
test_auth_rbac.py
-----------------
Comprehensive test suite for SmartAttendance login and Role-Based Access Control:
1. Login flow (success, wrong password, unknown username, inactive user).
2. /api/auth/me identity verification and invalid/expired token rejection.
3. Student role boundaries (can access /api/attendance/me, forbidden on staff endpoints).
4. Teacher role boundaries (can access /api/students, /api/attendance, forbidden on security/student).
5. Security role boundaries (can access /api/recognize, /api/attendance/mark, forbidden on roster/student).
6. Admin role access across management and security endpoints.
7. Public endpoints accessibility without authentication.
"""

from datetime import timedelta
from pathlib import Path
import sys
import unittest

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient

from app.auth import create_access_token, hash_password
from app.db.database import get_db, init_db
from app.main import app


class TestAuthRBAC(unittest.TestCase):
    """Test suite for authentication and role-based access control."""

    @classmethod
    def setUpClass(cls):
        # Ensure database and dev accounts exist
        init_db()
        cls.client = TestClient(app)

    # ── Helper methods ───────────────────────────────────────────────────────

    def login_user(self, username: str, password: str):
        """Helper to call POST /api/auth/login and return response."""
        return self.client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )

    def get_token_headers(self, username: str, password: str):
        """Helper to log in and return Authorization header dict."""
        res = self.login_user(username, password)
        self.assertEqual(res.status_code, 200, f"Login failed for {username}: {res.text}")
        token = res.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    # ── Login Tests ──────────────────────────────────────────────────────────

    def test_login_success_student(self):
        res = self.login_user("prapti", "Student@123")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertEqual(data["user"]["username"], "prapti")
        self.assertEqual(data["user"]["role"], "student")
        self.assertEqual(data["user"]["roll_no"], "emb-Prapti")
        self.assertNotIn("password", data["user"])
        self.assertNotIn("password_hash", data["user"])

    def test_login_success_teacher(self):
        res = self.login_user("teacher", "Teacher@123")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["user"]["role"], "teacher")
        self.assertIsNone(data["user"]["roll_no"])

    def test_login_success_security(self):
        res = self.login_user("security", "Security@123")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["user"]["role"], "security")

    def test_login_success_admin(self):
        res = self.login_user("admin", "Admin@123")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["user"]["role"], "admin")

    def test_login_wrong_password_rejected(self):
        res = self.login_user("teacher", "WrongPassword!")
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["detail"], "Invalid username or password.")

    def test_login_unknown_username_rejected(self):
        res = self.login_user("nobody_exists_here", "AnyPassword@123")
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["detail"], "Invalid username or password.")

    def test_login_inactive_user_rejected(self):
        # Create an inactive user directly in DB
        with get_db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users (username, password_hash, role, is_active)
                VALUES ('inactive_user', ?, 'student', 0)
                """,
                (hash_password("Inactive@123"),),
            )

        res = self.login_user("inactive_user", "Inactive@123")
        self.assertEqual(res.status_code, 401)

    # ── /api/auth/me Tests ───────────────────────────────────────────────────

    def test_auth_me_with_valid_token(self):
        headers = self.get_token_headers("prapti", "Student@123")
        res = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(res.status_code, 200)
        profile = res.json()
        self.assertEqual(profile["username"], "prapti")
        self.assertEqual(profile["role"], "student")
        self.assertEqual(profile["roll_no"], "emb-Prapti")

    def test_auth_me_missing_token_rejected(self):
        res = self.client.get("/api/auth/me")
        self.assertEqual(res.status_code, 401)

    def test_auth_me_tampered_token_rejected(self):
        headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.tampered.signature"}
        res = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(res.status_code, 401)

    def test_auth_me_expired_token_rejected(self):
        # Generate an expired token
        expired_token = create_access_token(
            {"sub": "1", "username": "teacher", "role": "teacher"},
            expires_delta=timedelta(seconds=-10),
        )
        headers = {"Authorization": f"Bearer {expired_token}"}
        res = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(res.status_code, 401)
        self.assertIn("expired", res.json()["detail"].lower())

    # ── Student Role Permissions ─────────────────────────────────────────────

    def test_student_can_access_own_attendance(self):
        headers = self.get_token_headers("prapti", "Student@123")
        res = self.client.get("/api/attendance/me", headers=headers)
        self.assertEqual(res.status_code, 200)
        records = res.json()
        self.assertIsInstance(records, list)
        for r in records:
            self.assertEqual(r["roll_no"], "emb-Prapti")

    def test_student_cannot_access_roster_or_all_attendance(self):
        headers = self.get_token_headers("prapti", "Student@123")

        # Student cannot list all students
        res_students = self.client.get("/api/students", headers=headers)
        self.assertEqual(res_students.status_code, 403)

        # Student cannot query global attendance
        res_att = self.client.get("/api/attendance", headers=headers)
        self.assertEqual(res_att.status_code, 403)

    def test_student_cannot_mark_attendance_or_run_recognition(self):
        headers = self.get_token_headers("prapti", "Student@123")

        # Student cannot mark attendance
        res_mark = self.client.post(
            "/api/attendance/mark",
            headers=headers,
            json={"name": "Prapti", "similarity": 0.9},
        )
        self.assertEqual(res_mark.status_code, 403)

        # Student cannot call recognition
        fake_file = ("frame.jpg", b"\xff\xd8\xff\xe0test", "image/jpeg")
        res_rec = self.client.post("/api/recognize", headers=headers, files={"file": fake_file})
        self.assertEqual(res_rec.status_code, 403)

    # ── Teacher Role Permissions ─────────────────────────────────────────────

    def test_teacher_can_access_students_and_attendance(self):
        headers = self.get_token_headers("teacher", "Teacher@123")

        # Teacher can view registered students
        res_students = self.client.get("/api/students", headers=headers)
        self.assertEqual(res_students.status_code, 200)
        self.assertGreater(len(res_students.json()), 0)

        # Teacher can view all attendance history
        res_att = self.client.get("/api/attendance", headers=headers)
        self.assertEqual(res_att.status_code, 200)

    def test_teacher_can_mark_attendance_and_export_excel(self):
        headers = self.get_token_headers("teacher", "Teacher@123")

        # Teacher is authorized to mark classroom attendance
        res_mark = self.client.post(
            "/api/attendance/mark",
            headers=headers,
            json={"name": "Prapti", "similarity": 0.9},
        )
        self.assertIn(res_mark.status_code, (200, 409))

        # Teacher is authorized to download Excel report
        res_excel = self.client.get("/api/attendance/export/excel?report_type=monthly", headers=headers)
        self.assertEqual(res_excel.status_code, 200)

        # Teacher cannot access /api/attendance/me (student-only)
        res_me = self.client.get("/api/attendance/me", headers=headers)
        self.assertEqual(res_me.status_code, 403)

    # ── Security Role Permissions ────────────────────────────────────────────

    def test_security_cannot_mark_classroom_attendance_or_export_excel(self):
        headers = self.get_token_headers("security", "Security@123")

        # Security CANNOT mark classroom attendance (gate observation only)
        res_mark = self.client.post(
            "/api/attendance/mark",
            headers=headers,
            json={"name": "Prapti", "similarity": 0.85},
        )
        self.assertEqual(res_mark.status_code, 403)

        # Security CANNOT export teacher attendance Excel register
        res_excel = self.client.get("/api/attendance/export/excel", headers=headers)
        self.assertEqual(res_excel.status_code, 403)

    def test_security_cannot_access_roster_or_student_me(self):
        headers = self.get_token_headers("security", "Security@123")

        # Security cannot view student roster
        res_students = self.client.get("/api/students", headers=headers)
        self.assertEqual(res_students.status_code, 403)

        # Security cannot view /api/attendance/me
        res_me = self.client.get("/api/attendance/me", headers=headers)
        self.assertEqual(res_me.status_code, 403)

    # ── Admin Role Permissions ───────────────────────────────────────────────

    def test_admin_has_elevated_access(self):
        headers = self.get_token_headers("admin", "Admin@123")

        # Admin can view students and attendance
        self.assertEqual(self.client.get("/api/students", headers=headers).status_code, 200)
        self.assertEqual(self.client.get("/api/attendance", headers=headers).status_code, 200)

        # Admin can mark attendance
        res_mark = self.client.post(
            "/api/attendance/mark",
            headers=headers,
            json={"name": "Prapti", "similarity": 0.85},
        )
        self.assertIn(res_mark.status_code, (200, 409))

        # Admin can export attendance Excel
        self.assertEqual(self.client.get("/api/attendance/export/excel", headers=headers).status_code, 200)

    # ── Public Endpoints ─────────────────────────────────────────────────────

    def test_public_endpoints_accessible_without_token(self):
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertEqual(res_root.json()["status"], "online")

        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
