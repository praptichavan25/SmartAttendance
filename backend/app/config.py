"""
config.py
---------
Centralised configuration for the SmartAttendance backend.
All settings are read from environment variables or a .env file.
Hard-coded production values are explicitly forbidden here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from the backend/ directory (one level up from app/)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ML_ROOT = _BACKEND_ROOT.parent / "ml"

load_dotenv(_BACKEND_ROOT / ".env")


# ── Server ────────────────────────────────────────────────────────────────────
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

# ── CORS ──────────────────────────────────────────────────────────────────────
# Only the local frontend origin is permitted.
CORS_ORIGIN: str = os.getenv("CORS_ORIGIN", "http://localhost:8080")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_BACKEND_ROOT / 'smartattendance.db'}",
)

# ── ML paths ──────────────────────────────────────────────────────────────────
# Defaults point at the existing ml/ directory layout.
# Override via .env for non-standard setups.
ML_EMBEDDINGS_PATH: Path = Path(
    os.getenv("ML_EMBEDDINGS_PATH", "") or (_ML_ROOT / "embeddings" / "embeddings.pkl")
)
ML_YUNET_MODEL_PATH: Path = Path(
    os.getenv("ML_YUNET_MODEL_PATH", "")
    or (_ML_ROOT / "models" / "face_detection_yunet_2023mar.onnx")
)
ML_SFACE_MODEL_PATH: Path = Path(
    os.getenv("ML_SFACE_MODEL_PATH", "")
    or (_ML_ROOT / "models" / "face_recognition_sface_2021dec.onnx")
)

# ── Recognition ───────────────────────────────────────────────────────────────
# Must match (or be stricter than) the value in ml/src/face_recognizer.py.
# Do NOT lower this without explicit approval.
COSINE_THRESHOLD: float = float(os.getenv("COSINE_THRESHOLD", "0.38"))

# ── Recognition Event Deduplication ──────────────────────────────────────────
# HOW LONG (seconds) the same continuously-visible identity is considered an
# "active track" — no new persistent events are created during this window.
RECOGNITION_ACTIVE_TRACK_SEC: float = float(os.getenv("RECOGNITION_ACTIVE_TRACK_SEC", "30.0"))

# HOW LONG (seconds) an identity must be ABSENT before a new recognition/alert
# event is created when they appear again ("new-visit" / re-entry cooldown).
RECOGNITION_REENTRY_SEC: float = float(os.getenv("RECOGNITION_REENTRY_SEC", "60.0"))

# HOW MANY FRAMES (of the same identity) must be seen before the first event
# is persisted — this avoids false positives from 1-frame detections.
RECOGNITION_MIN_CONFIRM_FRAMES: int = int(os.getenv("RECOGNITION_MIN_CONFIRM_FRAMES", "2"))

# HOW LONG (seconds) to wait before logging a new security_events row for an
# UNKNOWN person after the previous one.  Kept in sync with reentry cooldown.
UNKNOWN_ALERT_COOLDOWN_SEC: float = float(os.getenv("UNKNOWN_ALERT_COOLDOWN_SEC", "60.0"))

# ── Environment & Authentication ──────────────────────────────────────────────
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

_jwt_secret_env = os.getenv("JWT_SECRET_KEY", "").strip()

if ENVIRONMENT == "production":
    if not _jwt_secret_env:
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: JWT_SECRET_KEY must be set in production mode. "
            "Please configure JWT_SECRET_KEY in your environment or production .env file."
        )
    JWT_SECRET_KEY: str = _jwt_secret_env
else:
    # Development mode: use environment value if provided; otherwise use a development secret
    if _jwt_secret_env:
        JWT_SECRET_KEY: str = _jwt_secret_env
    else:
        JWT_SECRET_KEY: str = "dev-insecure-secret-key-change-in-production-smartattendance-2026"

