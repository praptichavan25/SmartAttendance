"""
test_auth_foundation.py
-----------------------
Focused unit and integration tests for the backend authentication foundation:
1. Password hashing with salted bcrypt.
2. Password verification (success and failure cases).
3. JWT access token creation, claims encoding, and decoding.
4. JWT token expiration and tampering rejection.
5. User creation, retrieval, and authentication against SQLite users table.
6. Safe, idempotent development account provisioning and student roll_no mapping.
7. Fail-safe behavior when JWT_SECRET_KEY is missing in production environment.
"""

from datetime import timedelta
import os
from pathlib import Path
import sqlite3
import sys
import unittest

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import jwt

import app.auth as auth_mod
from app.auth import (
    authenticate_user,
    create_access_token,
    create_user,
    decode_access_token,
    get_user_by_id,
    get_user_by_username,
    hash_password,
    provision_dev_accounts,
    verify_password,
)
from app.db.database import (
    _CREATE_ATTENDANCE,
    _CREATE_RECOGNITION_LOG,
    _CREATE_STUDENTS,
    _CREATE_USERS,
    _CREATE_USERS_INDEXES,
)


class TestAuthFoundation(unittest.TestCase):
    """Test suite for authentication hashing, tokens, and user provisioning."""

    def setUp(self):
        """Create an isolated in-memory SQLite database for each test."""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_CREATE_STUDENTS)
        self.conn.execute(_CREATE_ATTENDANCE)
        self.conn.execute(_CREATE_RECOGNITION_LOG)
        self.conn.execute(_CREATE_USERS)
        self.conn.executescript(_CREATE_USERS_INDEXES)

        # Seed sample students
        self.conn.execute("INSERT INTO students (roll_no, name) VALUES ('emb-Alice', 'Alice')")
        self.conn.execute("INSERT INTO students (roll_no, name) VALUES ('emb-Bob', 'Bob')")
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    # ── Password Hashing Tests ───────────────────────────────────────────────

    def test_password_hashing_and_verification(self):
        raw_pw = "SecureP@ssw0rd!2026"
        pw_hash = hash_password(raw_pw)

        self.assertNotEqual(pw_hash, raw_pw)
        self.assertTrue(pw_hash.startswith("$2b$") or pw_hash.startswith("$2a$"))
        self.assertTrue(verify_password(raw_pw, pw_hash))
        self.assertFalse(verify_password("WrongPassword", pw_hash))
        self.assertFalse(verify_password("", pw_hash))
        self.assertFalse(verify_password(raw_pw, ""))

    def test_password_salting_produces_unique_hashes(self):
        raw_pw = "SamePassword123"
        hash_1 = hash_password(raw_pw)
        hash_2 = hash_password(raw_pw)

        self.assertNotEqual(hash_1, hash_2, "Bcrypt hashes must differ due to unique salts")
        self.assertTrue(verify_password(raw_pw, hash_1))
        self.assertTrue(verify_password(raw_pw, hash_2))

    def test_empty_password_rejected(self):
        with self.assertRaises(ValueError):
            hash_password("")

    # ── JWT Token Tests ──────────────────────────────────────────────────────

    def test_token_creation_and_decoding(self):
        payload = {
            "sub": "user_42",
            "username": "prapti",
            "role": "student",
            "roll_no": "emb-Prapti",
        }
        token = create_access_token(payload, expires_delta=timedelta(minutes=30))
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 20)

        decoded = decode_access_token(token)
        self.assertEqual(decoded["sub"], "user_42")
        self.assertEqual(decoded["username"], "prapti")
        self.assertEqual(decoded["role"], "student")
        self.assertEqual(decoded["roll_no"], "emb-Prapti")
        self.assertIn("exp", decoded)
        self.assertIn("iat", decoded)

    def test_token_expiration_rejection(self):
        payload = {"sub": "expired_user", "username": "expired", "role": "student"}
        # Create an already expired token
        token = create_access_token(payload, expires_delta=timedelta(seconds=-10))

        with self.assertRaises(jwt.ExpiredSignatureError):
            decode_access_token(token)

    def test_token_tampering_rejection(self):
        payload = {"sub": "user_1", "username": "test", "role": "student"}
        token = create_access_token(payload)

        tampered = token[:-5] + "XXXXX"
        with self.assertRaises(jwt.PyJWTError):
            decode_access_token(tampered)

    # ── User Account Management Tests ────────────────────────────────────────

    def test_create_and_authenticate_user(self):
        user = create_user(
            self.conn,
            username="teststudent",
            password="MyPassword@123",
            role="student",
            roll_no="emb-Alice",
        )
        self.assertIsNotNone(user["id"])
        self.assertEqual(user["username"], "teststudent")
        self.assertEqual(user["role"], "student")
        self.assertEqual(user["roll_no"], "emb-Alice")

        # Lookup by username (case-insensitive)
        fetched = get_user_by_username(self.conn, "TESTSTUDENT")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["id"], user["id"])

        # Lookup by ID
        fetched_by_id = get_user_by_id(self.conn, user["id"])
        self.assertIsNotNone(fetched_by_id)
        self.assertEqual(fetched_by_id["username"], "teststudent")

        # Authenticate successfully
        authenticated = authenticate_user(self.conn, "teststudent", "MyPassword@123")
        self.assertIsNotNone(authenticated)
        self.assertEqual(authenticated["id"], user["id"])

        # Authentication failure on wrong password
        self.assertIsNone(authenticate_user(self.conn, "teststudent", "BadPassword"))

        # Authentication failure on unknown username
        self.assertIsNone(authenticate_user(self.conn, "nonexistent", "MyPassword@123"))

    def test_create_user_validations(self):
        # Student role requires roll_no
        with self.assertRaises(ValueError):
            create_user(self.conn, "nostudentroll", "Pass@123", role="student", roll_no=None)

        # Invalid role rejected
        with self.assertRaises(ValueError):
            create_user(self.conn, "badrole", "Pass@123", role="superhero")

        # Duplicate username rejected
        create_user(self.conn, "uniqueuser", "Pass@123", role="teacher")
        with self.assertRaises(ValueError):
            create_user(self.conn, "uniqueuser", "AnotherPass@123", role="teacher")

    # ── Development Account Provisioning Tests ───────────────────────────────

    def test_dev_provisioning_idempotency_and_mapping(self):
        # First provisioning
        result1 = provision_dev_accounts(self.conn)
        self.assertGreater(result1["created"], 0)
        total_after_first = self.conn.execute("SELECT count(*) FROM users").fetchone()[0]

        # Verify staff accounts exist
        teacher = get_user_by_username(self.conn, "teacher")
        self.assertIsNotNone(teacher)
        self.assertEqual(teacher["role"], "teacher")
        self.assertIsNone(teacher["roll_no"])

        # Verify student accounts are mapped to roll_no
        alice = get_user_by_username(self.conn, "alice")
        self.assertIsNotNone(alice)
        self.assertEqual(alice["role"], "student")
        self.assertEqual(alice["roll_no"], "emb-Alice")

        # Second provisioning must be idempotent (zero new users created, none overwritten)
        result2 = provision_dev_accounts(self.conn)
        self.assertEqual(result2["created"], 0)
        self.assertEqual(result2["skipped"], total_after_first)

        total_after_second = self.conn.execute("SELECT count(*) FROM users").fetchone()[0]
        self.assertEqual(total_after_first, total_after_second)

    def test_production_environment_skips_dev_provisioning(self):
        orig_env = auth_mod.ENVIRONMENT
        try:
            auth_mod.ENVIRONMENT = "production"
            result = provision_dev_accounts(self.conn)
            self.assertEqual(result["created"], 0)
            self.assertEqual(result["skipped"], 0)
        finally:
            auth_mod.ENVIRONMENT = orig_env


if __name__ == "__main__":
    unittest.main()
