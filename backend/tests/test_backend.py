"""
test_backend.py
---------------
Automated tests for the SmartAttendance FastAPI backend:
1. Health check (/api/health)
2. Database initialization and student seeding (/api/students)
3. Attendance marking (/api/attendance/mark) and duplicate rejection
4. Attendance listing (/api/attendance)
5. Face recognition endpoint (/api/recognize) with a real test image
"""

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import get_db, init_db
from app.auth import create_access_token

# Ensure tables are initialized for testing
init_db()
client = TestClient(app)


def get_token_headers(username: str, role: str) -> dict:
    """Generate authenticated bearer header for test user."""
    with get_db() as conn:
        user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        assert user, f"User '{username}' must exist in database"
        user_id = user["id"]
    token = create_access_token({"sub": str(user_id), "role": role})
    return {"Authorization": f"Bearer {token}"}


def test_root_endpoint():
    print("--> Testing GET /")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    print("    [PASS] Root endpoint returned status 'online'")


def test_health_endpoint():
    print("--> Testing GET /api/health")
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ml_ready"] is True
    assert data["database_ready"] is True
    assert data["total_students"] > 0
    print(f"    [PASS] Health check OK. Registered students in DB: {data['total_students']}")


def test_students_endpoint():
    print("--> Testing GET /api/students (authenticated as teacher)")
    headers = get_token_headers("teacher", "teacher")
    response = client.get("/api/students", headers=headers)
    assert response.status_code == 200
    students = response.json()
    assert isinstance(students, list)
    assert len(students) > 0
    names = [s["name"] for s in students]
    print(f"    [PASS] Students endpoint returned {len(students)} students: {', '.join(names)}")


def test_attendance_mark_and_duplicate():
    print("--> Testing POST /api/attendance/mark and duplicate prevention (authenticated as teacher)")
    headers = get_token_headers("teacher", "teacher")
    # Mark attendance for Prapti
    res = client.post("/api/attendance/mark", json={"name": "Prapti", "similarity": 0.85}, headers=headers)
    assert res.status_code in (200, 409), f"Unexpected status: {res.status_code}"
    print(f"    Mark response: {res.json()}")

    # Second mark attempt must return 409 Conflict
    dup_res = client.post("/api/attendance/mark", json={"name": "Prapti", "similarity": 0.88}, headers=headers)
    assert dup_res.status_code == 409
    dup_data = dup_res.json()
    assert dup_data["status"] == "duplicate"
    print(f"    [PASS] Duplicate attendance prevented as expected: {dup_data['message']}")


def test_attendance_list():
    print("--> Testing GET /api/attendance (authenticated as teacher)")
    headers = get_token_headers("teacher", "teacher")
    res = client.get("/api/attendance", headers=headers)
    assert res.status_code == 200
    records = res.json()
    assert isinstance(records, list)
    assert len(records) > 0
    print(f"    [PASS] Attendance endpoint returned {len(records)} records")


def test_recognize_invalid_image():
    print("--> Testing POST /api/recognize with invalid image (authenticated as security)")
    headers = get_token_headers("security", "security")
    fake_file = ("test.txt", b"not an image", "text/plain")
    res = client.post("/api/recognize", files={"file": fake_file}, headers=headers)
    assert res.status_code == 400
    print(f"    [PASS] Invalid image rejected with 400: {res.json()['detail']}")


def test_recognize_real_image():
    print("--> Testing POST /api/recognize with real test photo (authenticated as security)")
    headers = get_token_headers("security", "security")
    test_img_path = backend_dir.parent / "ml" / "dataset" / "test" / "registered" / "Prapti" / "test_photo.jpg"
    assert test_img_path.exists(), f"Test photo missing at {test_img_path}"

    with open(test_img_path, "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/recognize",
        files={"file": ("prapti_test.jpg", file_bytes, "image/jpeg")},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    print("    Recognition response:", data)
    assert data["faces_detected"] >= 1
    face_names = [f["name"] for f in data["results"]]
    assert "Prapti" in face_names
    prapti_face = next(f for f in data["results"] if f["name"] == "Prapti")
    assert prapti_face["status"] == "RECOGNISED"
    assert prapti_face["similarity"] >= 0.38
    print(f"    [PASS] Face recognized successfully as '{prapti_face['name']}' with similarity {prapti_face['similarity']}")


if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    test_students_endpoint()
    test_attendance_mark_and_duplicate()
    test_attendance_list()
    test_recognize_invalid_image()
    test_recognize_real_image()
    print("\nALL TEST SUITE CHECKS PASSED SUCCESSFULLY!")
