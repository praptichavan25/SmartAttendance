"""
schemas.py
----------
Pydantic schemas for SmartAttendance API requests and responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class StudentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    roll_no: str
    name: str
    department: str
    class_name: str = Field(alias="class")
    registered_at: str


class AttendanceRecord(BaseModel):
    id: int
    roll_no: str
    name: Optional[str] = None
    department: Optional[str] = None
    date: str
    entry_time: str
    status: str
    similarity: Optional[float] = None
    marked_at: str


class MarkAttendanceRequest(BaseModel):
    roll_no: Optional[str] = None
    name: Optional[str] = None
    similarity: Optional[float] = None


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceRecognitionResult(BaseModel):
    name: str
    similarity: float
    status: str
    bbox: BoundingBox
    detection_score: float
    identity_category: str = "REGISTERED_STUDENT"  # REGISTERED_STUDENT | APPROVED_GUEST | UNKNOWN_PERSON | INCONCLUSIVE
    authorization_status: str = "AUTHORIZED"       # AUTHORIZED | UNAUTHORIZED | UNVERIFIED
    box_color: str = "green"                       # green | orange | red | yellow
    guest_id: Optional[str] = None
    valid_until: Optional[str] = None
    label: Optional[str] = None


class RecognitionResponse(BaseModel):
    faces_detected: int
    results: List[FaceRecognitionResult]


# ── Guest Management Schemas ─────────────────────────────────────────────────

class GuestRegisterRequest(BaseModel):
    full_name: str
    purpose: str
    host_to_meet: str
    phone: Optional[str] = ""
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    image_base64: Optional[str] = None


class GuestResponse(BaseModel):
    guest_id: str
    full_name: str
    purpose: str
    host_to_meet: str
    phone: str = ""
    valid_from: str
    valid_until: str
    status: str
    face_snapshot: Optional[str] = None
    approved_by: str
    created_at: str
    revoked_by: Optional[str] = None
    revoked_at: Optional[str] = None
    revocation_reason: Optional[str] = None


class GuestRevokeRequest(BaseModel):
    reason: Optional[str] = "Revoked by security"


# ── Security Alert & Event Schemas ───────────────────────────────────────────

class SecurityEventResponse(BaseModel):
    event_id: str
    timestamp: str
    event_type: str
    identity_category: str
    name: str
    similarity: float
    source_camera: str
    bbox: Optional[BoundingBox] = None
    snapshot: Optional[str] = None
    resolution_status: str
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution_notes: Optional[str] = None


class EventResolveRequest(BaseModel):
    resolution_status: str = "resolved"
    resolution_notes: Optional[str] = ""


class HealthResponse(BaseModel):
    status: str
    service: str
    ml_ready: bool
    database_ready: bool
    total_students: int


# ── Authentication Schemas ───────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class UserSafeProfile(BaseModel):
    id: int
    username: str
    role: str
    roll_no: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserSafeProfile

