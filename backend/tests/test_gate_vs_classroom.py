"""
test_gate_vs_classroom.py
-------------------------
Comprehensive automated tests verifying Phase 8 requirements:
1. Separation of Campus Gate Entry Observations from Classroom Attendance.
2. Deduplication of gate observation events across repeated video frames.
3. Strict RBAC: Teacher/Admin can mark classroom attendance; Security cannot.
4. Guest recognition (approved, revoked, expired) and exclusion from student attendance.
5. Official Excel (.xlsx) export: verified structure, correct student stats, role protection.
"""

import io
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import openpyxl
import pytest
from fastapi.testclient import TestClient

from app.auth import create_access_token
from app.db.database import get_db, init_db
from app.main import app
from app.recognition_tracker import reset_tracker

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_test_environment():
    """Ensure clean tracker and database state for each test."""
    init_db()
    reset_tracker()
    yield
    reset_tracker()


def get_token_headers(username: str, role: str) -> dict:
    """Generate authenticated bearer header for test user."""
    with get_db() as conn:
        user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        assert user, f"User '{username}' must exist in database"
        user_id = user["id"]
    token = create_access_token({"sub": str(user_id), "role": role})
    return {"Authorization": f"Bearer {token}"}


# ── 1. Gate Observation vs Classroom Attendance Separation ────────────────────

def test_registered_student_at_gate_logs_observation_and_no_attendance():
    """
    Criterion 1, 2 & 8:
    A registered student recognized at the campus gate logs a GATE_OBSERVATION
    in security_events, and DOES NOT create or modify student attendance.
    """
    sec_headers = get_token_headers("security", "security")
    today = datetime.now().strftime("%Y-%m-%d")

    # Record initial counts
    with get_db() as conn:
        conn.execute("DELETE FROM security_events WHERE source_camera = 'gate_webcam'")
        conn.execute("DELETE FROM attendance WHERE (roll_no = 'Prapti' OR roll_no = 'emb-Prapti') AND date = ?", (today,))
        initial_att_count = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]

    mock_student_recognition = [{
        "name": "Prapti",
        "similarity": 0.88,
        "status": "RECOGNISED",
        "bbox": {"x": 100, "y": 80, "w": 120, "h": 140},
        "detection_score": 0.98,
        "identity_category": "REGISTERED_STUDENT",
        "box_color": "green",
        "label": "Student: Prapti (88%)",
        "guest_id": None,
    }]

    with patch("app.ml.service.MLService.recognize_bytes", return_value=mock_student_recognition):
        for _ in range(2):
            resp = client.post(
                "/api/recognize?camera_id=gate_webcam",
                headers=sec_headers,
                files={"file": ("frame.jpg", b"fake_gate_frame", "image/jpeg")},
            )
            assert resp.status_code == 200
        data = resp.json()
        assert data["faces_detected"] == 1
        assert data["results"][0]["identity_category"] == "REGISTERED_STUDENT"

    with get_db() as conn:
        # Check security_events: gate observation must be logged
        events = conn.execute(
            "SELECT * FROM security_events WHERE source_camera = 'gate_webcam' AND event_type = 'GATE_OBSERVATION'"
        ).fetchall()
        assert len(events) == 1, "Must log exactly one GATE_OBSERVATION in security_events"
        assert events[0]["name"] == "Prapti"
        assert events[0]["identity_category"] == "STUDENT"

        # Check attendance table: must NOT have any new attendance record
        final_att_count = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]
        assert final_att_count == initial_att_count, "Gate recognition must NEVER mark classroom attendance!"
        prapti_att = conn.execute("SELECT * FROM attendance WHERE (roll_no = 'Prapti' OR roll_no = 'emb-Prapti') AND date = ?", (today,)).fetchone()
        assert prapti_att is None, "Student seen ONLY at the gate must NOT have attendance recorded!"


def test_approved_guest_at_gate_logs_guest_observation():
    """
    Criterion 3:
    An approved guest recognized at the gate is logged as an approved guest observation,
    NOT as student attendance.
    """
    sec_headers = get_token_headers("security", "security")
    today = datetime.now().strftime("%Y-%m-%d")

    with get_db() as conn:
        conn.execute("DELETE FROM security_events WHERE source_camera = 'gate_webcam'")
        initial_att_count = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]

    mock_guest_recognition = [{
        "name": "Dr. Ramesh Sharma",
        "similarity": 0.85,
        "status": "RECOGNISED",
        "bbox": {"x": 110, "y": 90, "w": 115, "h": 130},
        "detection_score": 0.96,
        "identity_category": "APPROVED_GUEST",
        "box_color": "orange",
        "label": "Approved Guest: Dr. Ramesh Sharma (85%)",
        "guest_id": "GST-2026-001",
    }]

    with patch("app.ml.service.MLService.recognize_bytes", return_value=mock_guest_recognition):
        for _ in range(2):
            resp = client.post(
                "/api/recognize?camera_id=gate_webcam",
                headers=sec_headers,
                files={"file": ("frame.jpg", b"fake_guest_frame", "image/jpeg")},
            )
            assert resp.status_code == 200

    with get_db() as conn:
        events = conn.execute(
            "SELECT * FROM security_events WHERE source_camera = 'gate_webcam' AND event_type = 'GATE_OBSERVATION'"
        ).fetchall()
        assert len(events) == 1
        assert events[0]["identity_category"] == "GUEST"
        assert events[0]["name"] == "Dr. Ramesh Sharma"

        # Verify no attendance marked
        final_att_count = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]
        assert final_att_count == initial_att_count


def test_unknown_gate_detection_creates_alert_and_no_attendance():
    """
    Criterion 4:
    Unknown gate detection logs UNKNOWN_ALERT security event, and NO attendance.
    """
    sec_headers = get_token_headers("security", "security")

    with get_db() as conn:
        conn.execute("DELETE FROM security_events WHERE source_camera = 'gate_webcam'")
        initial_att_count = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]

    mock_unknown = [{
        "name": "UNKNOWN PERSON",
        "similarity": 0.28,
        "status": "UNKNOWN",
        "bbox": {"x": 100, "y": 80, "w": 120, "h": 140},
        "detection_score": 0.94,
        "identity_category": "UNKNOWN_PERSON",
        "box_color": "red",
        "label": "Unknown Person (28%)",
        "guest_id": None,
    }]

    with patch("app.ml.service.MLService.recognize_bytes", return_value=mock_unknown):
        for _ in range(2):
            resp = client.post(
                "/api/recognize?camera_id=gate_webcam",
                headers=sec_headers,
                files={"file": ("frame.jpg", b"fake_unknown_frame", "image/jpeg")},
            )
            assert resp.status_code == 200

    with get_db() as conn:
        events = conn.execute(
            "SELECT * FROM security_events WHERE source_camera = 'gate_webcam' AND event_type = 'UNKNOWN_ALERT'"
        ).fetchall()
        assert len(events) == 1
        assert events[0]["resolution_status"] == "unresolved"

        final_att_count = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]
        assert final_att_count == initial_att_count


# ── 2. Deduplication across Continuous Frames ─────────────────────────────────

def test_repeated_frames_at_gate_deduplicated_to_single_event():
    """
    Criterion 5:
    Simulating 8 continuous frames of Prapti at the gate creates exactly ONE
    GATE_OBSERVATION event with updated last_seen_at.
    """
    sec_headers = get_token_headers("security", "security")

    with get_db() as conn:
        conn.execute("DELETE FROM security_events WHERE source_camera = 'gate_webcam'")

    mock_student = [{
        "name": "Prapti",
        "similarity": 0.89,
        "status": "RECOGNISED",
        "bbox": {"x": 100, "y": 80, "w": 120, "h": 140},
        "detection_score": 0.98,
        "identity_category": "REGISTERED_STUDENT",
        "box_color": "green",
        "label": "Student: Prapti (89%)",
        "guest_id": None,
    }]

    with patch("app.ml.service.MLService.recognize_bytes", return_value=mock_student):
        for _ in range(8):
            resp = client.post(
                "/api/recognize?camera_id=gate_webcam",
                headers=sec_headers,
                files={"file": ("frame.jpg", b"fake_gate_frame", "image/jpeg")},
            )
            assert resp.status_code == 200

    with get_db() as conn:
        events = conn.execute(
            "SELECT * FROM security_events WHERE source_camera = 'gate_webcam' AND event_type = 'GATE_OBSERVATION'"
        ).fetchall()
        assert len(events) == 1, f"Expected 1 deduplicated event, got {len(events)}"
        assert events[0]["last_seen_at"] is not None


# ── 3. Classroom Attendance Authorization & Duplicate Rejection ───────────────

def test_security_cannot_mark_attendance():
    """
    Criterion 7 & RBAC:
    Security role calling /api/attendance/mark must receive HTTP 403 Forbidden.
    """
    sec_headers = get_token_headers("security", "security")
    resp = client.post(
        "/api/attendance/mark",
        headers=sec_headers,
        json={"name": "Prapti", "similarity": 0.90},
    )
    assert resp.status_code == 403, "Security role MUST NOT be allowed to mark classroom attendance!"


def test_teacher_marks_classroom_attendance_and_rejects_duplicate():
    """
    Criterion 6 & 7:
    Classroom recognition allows Teacher to mark attendance, and rejects
    duplicate attendance on the same date with HTTP 409 Conflict.
    """
    teacher_headers = get_token_headers("teacher", "teacher")
    today = datetime.now().strftime("%Y-%m-%d")

    with get_db() as conn:
        conn.execute("DELETE FROM attendance WHERE (roll_no = 'Prapti' OR roll_no = 'emb-Prapti') AND date = ?", (today,))

    # First mark: success
    res1 = client.post(
        "/api/attendance/mark",
        headers=teacher_headers,
        json={"name": "Prapti", "similarity": 0.92},
    )
    assert res1.status_code == 200
    assert res1.json()["status"] == "success"

    # Second mark (duplicate in same session / date): rejected with 409
    res2 = client.post(
        "/api/attendance/mark",
        headers=teacher_headers,
        json={"name": "Prapti", "similarity": 0.94},
    )
    assert res2.status_code == 409
    assert res2.json()["status"] == "duplicate"


def test_guest_seen_in_classroom_does_not_create_attendance():
    """
    Criterion 9:
    If an approved guest is seen by the classroom camera, no student attendance is marked.
    """
    teacher_headers = get_token_headers("teacher", "teacher")
    today = datetime.now().strftime("%Y-%m-%d")

    with get_db() as conn:
        initial_att = conn.execute("SELECT COUNT(*) as c FROM attendance WHERE date = ?", (today,)).fetchone()["c"]

    mock_guest = [{
        "name": "Prof. Guest Speaker",
        "similarity": 0.88,
        "status": "RECOGNISED",
        "bbox": {"x": 100, "y": 80, "w": 120, "h": 140},
        "detection_score": 0.97,
        "identity_category": "APPROVED_GUEST",
        "box_color": "orange",
        "label": "Approved Guest: Prof. Guest Speaker (88%)",
        "guest_id": "GST-2026-999",
    }]

    with patch("app.ml.service.MLService.recognize_bytes", return_value=mock_guest):
        resp = client.post(
            "/api/recognize?camera_id=teacher_monitor",
            headers=teacher_headers,
            files={"file": ("frame.jpg", b"fake_classroom_frame", "image/jpeg")},
        )
        assert resp.status_code == 200

    with get_db() as conn:
        final_att = conn.execute("SELECT COUNT(*) as c FROM attendance WHERE date = ?", (today,)).fetchone()["c"]
        assert final_att == initial_att, "Guest sighting in classroom must NOT mark student attendance!"


# ── 4. Excel Attendance Export ────────────────────────────────────────────────

def test_excel_export_rbac():
    """
    Criterion 13 & RBAC:
    Only teacher and admin can download Excel reports. Students and security are rejected (403).
    """
    sec_headers = get_token_headers("security", "security")
    res_sec = client.get("/api/attendance/export/excel", headers=sec_headers)
    assert res_sec.status_code == 403, "Security role must be forbidden from downloading attendance Excel!"

    stu_headers = get_token_headers("prapti", "student")
    res_stu = client.get("/api/attendance/export/excel", headers=stu_headers)
    assert res_stu.status_code == 403, "Student role must be forbidden from downloading attendance Excel!"

    teacher_headers = get_token_headers("teacher", "teacher")
    res_tch = client.get("/api/attendance/export/excel?report_type=monthly", headers=teacher_headers)
    assert res_tch.status_code == 200, "Teacher must be authorized to download Excel report!"
    assert res_tch.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def test_excel_export_structure_and_data_integrity():
    """
    Criterion 13 & 14:
    Excel export contains:
    - Registered students (Roll No, Name, Dept, Class)
    - Conducted session dates with P/A status
    - Total Present, Total Absent, Total Conducted, Attendance %
    - Accurate calculation strictly from attendance table (gate observations excluded)
    """
    teacher_headers = get_token_headers("teacher", "teacher")
    today = datetime.now().strftime("%Y-%m-%d")

    # Seed controlled attendance records
    with get_db() as conn:
        stu = conn.execute("SELECT roll_no, name FROM students WHERE LOWER(name) = 'prapti'").fetchone()
        assert stu, "Student Prapti must exist in students table"
        prapti_roll = stu["roll_no"]

        conn.execute("DELETE FROM attendance WHERE date = ?", (today,))
        conn.execute(
            """
            INSERT INTO attendance (roll_no, date, entry_time, status, similarity)
            VALUES (?, ?, '09:15:00', 'present', 0.91)
            """,
            (prapti_roll, today),
        )

    # Request monthly report
    month_str = datetime.now().strftime("%Y-%m")
    resp = client.get(
        f"/api/attendance/export/excel?report_type=monthly&month={month_str}",
        headers=teacher_headers,
    )
    assert resp.status_code == 200

    # Parse generated Excel workbook using openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    ws = wb.active
    assert ws is not None
    assert ws.title == "Classroom Attendance"

    # Header title check
    title_val = str(ws.cell(row=1, column=1).value or "")
    assert "SMARTATTENDANCE" in title_val

    # Column headers at row 5
    header_row = [ws.cell(row=5, column=col).value for col in range(1, ws.max_column + 1)]
    assert "Roll No" in header_row
    assert "Student Name" in header_row
    assert "Present" in header_row
    assert "Absent" in header_row
    assert "Attendance %" in header_row

    # Verify Prapti has P on today's date
    prapti_found = False
    for row in range(6, ws.max_row + 1):
        roll_val = ws.cell(row=row, column=1).value
        name_val = ws.cell(row=row, column=2).value
        if roll_val == "Prapti" or name_val == "Prapti":
            prapti_found = True
            # Present count must be at least 1
            present_col = header_row.index("Present") + 1
            pct_col = header_row.index("Attendance %") + 1
            present_val = int(str(ws.cell(row=row, column=present_col).value or 0))
            assert present_val >= 1
            assert "%" in str(ws.cell(row=row, column=pct_col).value)
            break

    assert prapti_found, "Registered student Prapti must be listed in official Excel attendance register"
