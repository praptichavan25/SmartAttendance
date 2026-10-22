"""
auth.py
-------
Authentication foundation module for SmartAttendance backend.

Features:
- Salted password hashing and verification using bcrypt.
- Signed, expiring JWT access token creation and decoding using PyJWT.
- User account creation, lookup, and authentication against SQLite users table.
- Safe development account provisioning that preserves existing passwords and student mappings.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import sqlite3
from typing import Any, Dict, Optional

import bcrypt
import jwt

from app.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ENVIRONMENT,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
)

logger = logging.getLogger(__name__)

# Allowed system roles
VALID_ROLES = {"student", "teacher", "security", "admin"}


# ── Password Hashing & Verification ──────────────────────────────────────────

def hash_password(password: str) -> str:
    """
    Hash a plaintext password using salted bcrypt with work factor 12.

    Bcrypt automatically generates a unique 128-bit salt and bundles the salt,
    cost factor, and hash into a standard 60-character modular crypt format.
    """
    if not password:
        raise ValueError("Password cannot be empty.")

    pw_bytes = password.encode("utf-8")
    # Bcrypt maximum input length is 72 bytes
    if len(pw_bytes) > 72:
        pw_bytes = pw_bytes[:72]

    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(pw_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.
    Returns True if valid, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False

    pw_bytes = plain_password.encode("utf-8")
    if len(pw_bytes) > 72:
        pw_bytes = pw_bytes[:72]

    try:
        return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))
    except Exception as exc:
        logger.warning("Error verifying password hash: %s", exc)
        return False


# ── JWT Token Management ─────────────────────────────────────────────────────

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a cryptographically signed, expiring JWT access token.

    Payload claims:
    - sub: User ID or primary identifier
    - username: Unique username
    - role: System role (student, teacher, security, admin)
    - roll_no: Student roll number if role == student, else None
    - iat: Issued at timestamp (UTC)
    - exp: Expiration timestamp (UTC)
    """
    to_encode = data.copy()
    now_utc = datetime.now(timezone.utc)
    expire = now_utc + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    to_encode.update({
        "iat": now_utc,
        "exp": expire,
    })

    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a signed JWT access token.

    Raises:
    - jwt.ExpiredSignatureError: Token has passed its expiration time.
    - jwt.InvalidTokenError: Token signature is invalid or payload is corrupted.
    """
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


# ── User Account Management ──────────────────────────────────────────────────

def create_user(
    conn: sqlite3.Connection,
    username: str,
    password: str,
    role: str,
    roll_no: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new user account with a salted password hash.
    Raises ValueError on invalid role or duplicate username.
    """
    username_clean = username.strip().lower()
    if not username_clean:
        raise ValueError("Username cannot be empty.")

    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of: {VALID_ROLES}")

    if role == "student" and not roll_no:
        raise ValueError("Student accounts must specify a valid roll_no.")

    # Check for existing username
    cursor = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (username_clean,),
    )
    if cursor.fetchone():
        raise ValueError(f"Username '{username_clean}' already exists.")

    pw_hash = hash_password(password)
    cursor = conn.execute(
        """
        INSERT INTO users (username, password_hash, role, roll_no)
        VALUES (?, ?, ?, ?)
        """,
        (username_clean, pw_hash, role, roll_no),
    )
    user_id = cursor.lastrowid

    return {
        "id": user_id,
        "username": username_clean,
        "role": role,
        "roll_no": roll_no,
    }


def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[Dict[str, Any]]:
    """Retrieve user record by username."""
    cursor = conn.execute(
        """
        SELECT id, username, password_hash, role, roll_no, is_active, created_at
        FROM users
        WHERE username = ?
        """,
        (username.strip().lower(),),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return dict(row)


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve user record by user ID."""
    cursor = conn.execute(
        """
        SELECT id, username, password_hash, role, roll_no, is_active, created_at
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return dict(row)


def authenticate_user(
    conn: sqlite3.Connection,
    username: str,
    password: str,
) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user by username and plaintext password.
    Returns user dict if valid and active, None otherwise.
    """
    user = get_user_by_username(conn, username)
    if not user:
        return None

    if not user.get("is_active", 1):
        logger.warning("Attempted login to inactive account: %s", username)
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return user


# ── Development Account Provisioning ─────────────────────────────────────────

def provision_dev_accounts(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Safely provision development accounts for local prototype testing without
    resetting existing passwords on startup.

    Rules:
    - Skips completely if ENVIRONMENT == 'production'.
    - Checks whether each account already exists before inserting.
    - Never overwrites or resets existing passwords.
    - Maps student accounts to the actual roll_no values in the students table.
    """
    if ENVIRONMENT == "production":
        logger.info("ENVIRONMENT is 'production'; skipping dev account provisioning.")
        return {"created": 0, "skipped": 0}

    created = 0
    skipped = 0

    # 1. Staff development accounts
    staff_accounts = [
        ("teacher", "Teacher@123", "teacher", None),
        ("security", "Security@123", "security", None),
        ("admin", "Admin@123", "admin", None),
    ]

    for username, plain_pw, role, roll_no in staff_accounts:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username.lower(),),
        ).fetchone()

        if existing:
            skipped += 1
            continue

        pw_hash = hash_password(plain_pw)
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, roll_no)
            VALUES (?, ?, ?, ?)
            """,
            (username.lower(), pw_hash, role, roll_no),
        )
        created += 1

    # 2. Student development accounts mapped to existing rows in students table
    student_rows = conn.execute("SELECT roll_no, name FROM students").fetchall()
    for s in student_rows:
        roll_no = s["roll_no"]
        name = s["name"]
        username = name.strip().lower()

        # Check by username or roll_no
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ? OR roll_no = ?",
            (username, roll_no),
        ).fetchone()

        if existing:
            skipped += 1
            continue

        pw_hash = hash_password("Student@123")
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, roll_no)
            VALUES (?, ?, 'student', ?)
            """,
            (username, pw_hash, roll_no),
        )
        created += 1

    if created > 0:
        logger.info(
            "Provisioned %d development accounts (%d already existed and preserved).",
            created,
            skipped,
        )

    return {"created": created, "skipped": skipped}
