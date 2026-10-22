"""
database.py
-----------
SQLite database setup using Python's built-in sqlite3 module.

Schema
------
students        — registered student records (name, roll_no, department, class)
attendance      — one row per student per attendance date
                  duplicate entries for (roll_no, date) are prevented by a
                  UNIQUE constraint enforced at the DB level.

No face images or raw embeddings are stored in the database.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.config import DATABASE_URL

# Resolve the file path from the sqlite:/// URL
_DB_PATH = Path(DATABASE_URL.replace("sqlite:///", ""))

# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    roll_no     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL DEFAULT '',
    class       TEXT NOT NULL DEFAULT '',
    registered_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_ATTENDANCE = """
CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no     TEXT    NOT NULL REFERENCES students(roll_no),
    date        TEXT    NOT NULL,          -- ISO date YYYY-MM-DD
    entry_time  TEXT    NOT NULL,          -- ISO time HH:MM:SS
    status      TEXT    NOT NULL DEFAULT 'present'
                        CHECK(status IN ('present', 'absent')),
    similarity  REAL,                      -- cosine similarity from ML (nullable)
    session_id  TEXT,                      -- future: link to a recognition session
    marked_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(roll_no, date)                  -- one attendance record per student per day
);
"""

_CREATE_RECOGNITION_LOG = """
CREATE TABLE IF NOT EXISTS recognition_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    name            TEXT    NOT NULL,      -- matched name or 'UNKNOWN'
    similarity      REAL    NOT NULL,
    status          TEXT    NOT NULL,      -- 'RECOGNISED' | 'UNKNOWN' | 'NO_FACE'
    bbox_x          INTEGER,
    bbox_y          INTEGER,
    bbox_w          INTEGER,
    bbox_h          INTEGER,
    detection_score REAL,
    source          TEXT    DEFAULT 'api'  -- 'api' | 'live' for future use
);
"""

_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK(role IN ('student', 'teacher', 'security', 'admin')),
    roll_no       TEXT NULL REFERENCES students(roll_no) ON DELETE SET NULL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_USERS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_roll_no ON users(roll_no);
"""

_CREATE_GUESTS = """
CREATE TABLE IF NOT EXISTS guests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_id            TEXT UNIQUE NOT NULL,
    full_name           TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    host_to_meet        TEXT NOT NULL,
    phone               TEXT NOT NULL DEFAULT '',
    valid_from          TEXT NOT NULL,              -- ISO datetime YYYY-MM-DDTHH:MM:SS
    valid_until         TEXT NOT NULL,              -- ISO datetime YYYY-MM-DDTHH:MM:SS
    status              TEXT NOT NULL DEFAULT 'approved'
                        CHECK(status IN ('approved', 'revoked', 'expired')),
    face_embedding      BLOB NOT NULL,              -- 128-d float32 SFace embedding bytes
    face_snapshot       TEXT NOT NULL DEFAULT '',   -- data URL / thumbnail
    approved_by         TEXT NOT NULL,
    revoked_by          TEXT,
    revoked_at          TEXT,
    revocation_reason   TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_GUESTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_guests_guest_id ON guests(guest_id);
CREATE INDEX IF NOT EXISTS idx_guests_status ON guests(status);
CREATE INDEX IF NOT EXISTS idx_guests_valid_until ON guests(valid_until);
"""

_CREATE_SECURITY_EVENTS = """
CREATE TABLE IF NOT EXISTS security_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id            TEXT UNIQUE NOT NULL,
    timestamp           TEXT NOT NULL DEFAULT (datetime('now')),
    event_type          TEXT NOT NULL,              -- 'UNKNOWN_ALERT', 'GUEST_RECOGNIZED', 'STUDENT_RECOGNIZED', 'GUEST_REGISTERED', 'GUEST_REVOKED'
    identity_category   TEXT NOT NULL,              -- 'STUDENT', 'GUEST', 'UNKNOWN', 'INCONCLUSIVE'
    name                TEXT NOT NULL,
    similarity          REAL NOT NULL DEFAULT 0.0,
    source_camera       TEXT NOT NULL DEFAULT 'gate_webcam',
    bbox_x              INTEGER,
    bbox_y              INTEGER,
    bbox_w              INTEGER,
    bbox_h              INTEGER,
    snapshot            TEXT NOT NULL DEFAULT '',
    resolution_status   TEXT NOT NULL DEFAULT 'unresolved'
                        CHECK(resolution_status IN ('unresolved', 'investigating', 'resolved', 'registered_as_guest', 'dismissed')),
    resolved_by         TEXT,
    resolved_at         TEXT,
    resolution_notes    TEXT,
    -- deduplication fields (added in migration v2 -- backward compatible)
    track_key           TEXT,                       -- opaque key matching RecognitionTracker
    last_seen_at        TEXT                        -- ISO datetime of last frame containing this track
);
"""

_CREATE_SECURITY_EVENTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(event_type);
CREATE INDEX IF NOT EXISTS idx_security_events_time ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_status ON security_events(resolution_status);
CREATE INDEX IF NOT EXISTS idx_security_events_track_key ON security_events(track_key);
"""

# ── Additive migration for existing databases ─────────────────────────────────
# These statements silently succeed even if the columns already exist.
_MIGRATE_SECURITY_EVENTS_V2 = [
    "ALTER TABLE security_events ADD COLUMN track_key TEXT",
    "ALTER TABLE security_events ADD COLUMN last_seen_at TEXT",
]

# ── Connection helper ─────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Open a sqlite3 connection with row_factory set to Row."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # allows concurrent reads
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields an open connection and commits on exit."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialisation ────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they do not already exist, seed demo students, and provision dev accounts."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute(_CREATE_STUDENTS)
        conn.execute(_CREATE_ATTENDANCE)
        conn.execute(_CREATE_RECOGNITION_LOG)
        conn.execute(_CREATE_USERS)
        conn.executescript(_CREATE_USERS_INDEXES)
        conn.execute(_CREATE_GUESTS)
        conn.executescript(_CREATE_GUESTS_INDEXES)
        conn.execute(_CREATE_SECURITY_EVENTS)

        # Additive migration: add deduplication columns to existing databases before creating indexes.
        for stmt in _MIGRATE_SECURITY_EVENTS_V2:
            try:
                conn.execute(stmt)
            except Exception:
                pass  # Column already exists — nothing to do

        conn.executescript(_CREATE_SECURITY_EVENTS_INDEXES)

    _seed_registered_students()

    # Safely provision development accounts if running in development mode
    from app.auth import provision_dev_accounts
    with get_db() as conn:
        provision_dev_accounts(conn)



def _seed_registered_students() -> None:
    """
    Seed student rows from the names found in embeddings.pkl.

    This is called once at startup. It adds any registered face name that
    does not yet have a students row, using the embedding name as both
    roll_no and display name until real roll numbers are available.

    Roll numbers seeded here are prefixed with 'emb-' to distinguish them
    from proper enrollment records. A teacher or admin must update them later.
    """
    import pickle

    from app.config import ML_EMBEDDINGS_PATH

    if not ML_EMBEDDINGS_PATH.exists():
        return

    with open(ML_EMBEDDINGS_PATH, "rb") as f:
        embeddings: dict = pickle.load(f)

    with get_db() as conn:
        for name in embeddings.keys():
            roll_no = f"emb-{name}"
            conn.execute(
                """
                INSERT OR IGNORE INTO students (roll_no, name, department, class)
                VALUES (?, ?, ?, ?)
                """,
                (roll_no, name, "Unassigned", "Unassigned"),
            )
