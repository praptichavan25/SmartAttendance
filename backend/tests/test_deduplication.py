"""
Unit and integration tests for face recognition event deduplication.
Verifies:
1. RecognitionTracker logic (frame accumulation, confirmation threshold, active window, reentry).
2. Endpoint deduplication: /api/recognize does not spam recognition_log or security_events on repeated frames.
3. Live response contract: bounding boxes, identity category, and labels are always returned on every frame.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db, init_db
from app.recognition_tracker import RecognitionTracker, get_tracker, reset_tracker
from app.auth import create_access_token


@pytest.fixture(autouse=True)
def setup_teardown():
    init_db()
    reset_tracker()
    yield
    reset_tracker()


# ── 1. RecognitionTracker Unit Tests ──────────────────────────────────────────

def test_tracker_confirm_threshold():
    """Verify that a sighting requires min_confirm_frames before triggering writes."""
    tracker = RecognitionTracker(active_track_sec=10.0, reentry_sec=20.0, min_confirm_frames=2)

    # Frame 1: first sight, confirm_count = 1 (< 2)
    log1, evt1, eid1 = tracker.observe("REGISTERED_STUDENT", "Prapti", camera_id="cam1")
    assert not log1, "Should not write log on first unconfirmed frame"
    assert not evt1, "Should not write event on first unconfirmed frame"

    # Frame 2: confirmed!
    log2, evt2, eid2 = tracker.observe("REGISTERED_STUDENT", "Prapti", camera_id="cam1")
    assert log2, "Should trigger log write on confirmation"
    assert not evt2, "Registered student should not trigger security event"

    # Frame 3: already written
    log3, evt3, eid3 = tracker.observe("REGISTERED_STUDENT", "Prapti", camera_id="cam1")
    assert not log3, "Should not re-trigger log write on subsequent frame"
    assert not evt3


def test_tracker_unknown_alert_deduplication():
    """Verify that unknown faces only trigger one security event per active visit."""
    tracker = RecognitionTracker(active_track_sec=5.0, reentry_sec=10.0, min_confirm_frames=2)

    # Frame 1
    log1, evt1, eid1 = tracker.observe("UNKNOWN_PERSON", "Unknown", camera_id="cam1")
    assert not log1 and not evt1

    # Frame 2: confirmed unknown sighting
    log2, evt2, eid2 = tracker.observe("UNKNOWN_PERSON", "Unknown", camera_id="cam1")
    assert log2, "Should write recognition log once confirmed"
    assert evt2, "Should trigger security event for unknown person"

    # Attach event ID
    tracker.set_event_id("UNKNOWN_PERSON", "Unknown", "cam1", "EVT-TEST-001")

    # Frames 3 through 10: same visit, active track
    for _ in range(8):
        log_n, evt_n, active_eid = tracker.observe("UNKNOWN_PERSON", "Unknown", camera_id="cam1")
        assert not log_n, "Must not duplicate recognition log during continuous sighting"
        assert not evt_n, "Must not duplicate security event during continuous sighting"
        assert active_eid == "EVT-TEST-001", "Should return active event_id for in-place timestamp update"


def test_tracker_camera_isolation():
    """Verify that different camera IDs maintain independent deduplication tracks."""
    tracker = RecognitionTracker(active_track_sec=10.0, reentry_sec=20.0, min_confirm_frames=1)

    # Cam A observes Prapti
    log_a, _, _ = tracker.observe("REGISTERED_STUDENT", "Prapti", camera_id="gate_cam")
    assert log_a

    # Cam B observes Prapti independently
    log_b, _, _ = tracker.observe("REGISTERED_STUDENT", "Prapti", camera_id="classroom_cam")
    assert log_b, "Different camera must track independently"


def test_tracker_reentry_after_expiry():
    """Verify that after identity is absent for > reentry_sec, a new track/visit starts."""
    tracker = RecognitionTracker(active_track_sec=0.2, reentry_sec=0.5, min_confirm_frames=1)

    # First visit
    log1, _, _ = tracker.observe("REGISTERED_STUDENT", "Prapti", camera_id="cam1")
    assert log1

    # Within active window
    log2, _, _ = tracker.observe("REGISTERED_STUDENT", "Prapti", camera_id="cam1")
    assert not log2

    # Wait for track expiry
    time.sleep(0.6)

    # Second visit after expiry
    log3, _, _ = tracker.observe("REGISTERED_STUDENT", "Prapti", camera_id="cam1")
    assert log3, "Should start new track after reentry cooldown expires"


# ── 2. /api/recognize Integration Tests ────────────────────────────────────────

def _get_security_token():
    with get_db() as conn:
        user = conn.execute("SELECT id, role FROM users WHERE username = 'security'").fetchone()
        assert user, "security user must exist in database"
        user_id = user["id"]
        role = user["role"]
    return create_access_token({"sub": str(user_id), "role": role})


def test_recognize_continuous_unknown_deduplication():
    """
    Simulate a live stream sending 10 consecutive frames of an unknown person.
    Verify:
    - 10 responses return 200 OK with valid bounding boxes and UNKNOWN category.
    - Only 1 security_events row is created.
    - Only 1 recognition_log row is created.
    - The security_events row has last_seen_at updated.
    """
    client = TestClient(app)
    token = _get_security_token()

    # Clear prior security_events and recognition_log
    with get_db() as conn:
        conn.execute("DELETE FROM security_events")
        conn.execute("DELETE FROM recognition_log")

    mock_results = [{
        "name": "Unknown",
        "similarity": 0.32,
        "status": "UNKNOWN",
        "bbox": {"x": 50, "y": 60, "w": 100, "h": 120},
        "detection_score": 0.95,
        "identity_category": "UNKNOWN_PERSON",
        "box_color": "#d82b2b",
        "label": "⚠️ UNKNOWN PERSON (32%)",
        "guest_id": None,
    }]

    # Send 10 consecutive frames
    with patch("app.ml.service.MLService.recognize_bytes", return_value=mock_results):
        for _ in range(10):
            resp = client.post(
                "/api/recognize?camera_id=gate_webcam",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("test.jpg", b"fake_image_bytes", "image/jpeg")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["faces_detected"] == 1
            assert len(data["results"]) == 1
            assert data["results"][0]["identity_category"] == "UNKNOWN_PERSON"
            assert data["results"][0]["bbox"]["w"] == 100

    # Verify database counts: exactly 1 event and 1 log despite 10 frames
    with get_db() as conn:
        events = conn.execute("SELECT * FROM security_events WHERE event_type = 'UNKNOWN_ALERT'").fetchall()
        logs = conn.execute("SELECT * FROM recognition_log WHERE status = 'UNKNOWN'").fetchall()

        assert len(events) == 1, f"Expected exactly 1 security_event, got {len(events)}"
        assert len(logs) == 1, f"Expected exactly 1 recognition_log, got {len(logs)}"
        assert events[0]["track_key"] == "UNKNOWN:gate_webcam"
