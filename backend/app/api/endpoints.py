"""
endpoints.py
------------
FastAPI router implementing all required SmartAttendance endpoints:
- POST /api/auth/login      (Public: login and receive JWT)
- GET  /api/auth/me         (Authenticated: get current safe profile)
- GET  /api/health          (Public: system status & health check)
- GET  /api/students        (Protected: teacher, admin only)
- GET  /api/attendance      (Protected: teacher, admin only)
- GET  /api/attendance/me   (Protected: student only, linked to current_user roll_no)
- POST /api/attendance/mark (Protected: security, admin only)
- POST /api/recognize       (Protected: security, admin only)
"""

import asyncio
import base64
import calendar
from datetime import datetime
import io
import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from app.api.dependencies import get_current_user, require_roles
from app.api.schemas import (
    AttendanceRecord,
    BoundingBox,
    EventResolveRequest,
    FaceRecognitionResult,
    GuestRegisterRequest,
    GuestResponse,
    GuestRevokeRequest,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    MarkAttendanceRequest,
    RecognitionResponse,
    SecurityEventResponse,
    StudentResponse,
    UserSafeProfile,
)
from app.auth import authenticate_user, create_access_token
from app.db.database import get_db
from app.ml.service import MLService
from app.recognition_tracker import get_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["SmartAttendance"])

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/octet-stream",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit


# ── Authentication Endpoints ─────────────────────────────────────────────────

@router.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    """
    Authenticate user credentials (username and password) using bcrypt.
    Rejects invalid credentials or inactive accounts with HTTP 401 and a generic error message.
    Returns signed access token and safe user profile. Never returns password hashes or salts.
    """
    username = payload.username.strip().lower()
    with get_db() as conn:
        user = authenticate_user(conn, username, payload.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_claims = {
            "sub": str(user["id"]),
            "username": user["username"],
            "role": user["role"],
            "roll_no": user.get("roll_no"),
        }
        access_token = create_access_token(token_claims)

        user_profile = UserSafeProfile(
            id=user["id"],
            username=user["username"],
            role=user["role"],
            roll_no=user.get("roll_no"),
            is_active=bool(user.get("is_active", 1)),
            created_at=user.get("created_at"),
        )

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_profile,
        )


@router.get("/auth/me", response_model=UserSafeProfile)
def get_me(current_user: dict = Depends(get_current_user)):
    """
    Return the authenticated user's safe profile.
    Requires a valid bearer token.
    """
    return UserSafeProfile(
        id=current_user["id"],
        username=current_user["username"],
        role=current_user["role"],
        roll_no=current_user.get("roll_no"),
        is_active=bool(current_user.get("is_active", 1)),
        created_at=current_user.get("created_at"),
    )


# ── Health Endpoint (Public) ─────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report system health, ML model status, and database connectivity."""
    ml_ready = MLService.get_instance().is_ready
    db_ready = False
    total_students = 0

    try:
        with get_db() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM students")
            total_students = cursor.fetchone()[0]
            db_ready = True
    except Exception as exc:
        logger.error("Health check database error: %s", exc)

    return HealthResponse(
        status="ok" if (ml_ready and db_ready) else "degraded",
        service="SmartAttendance Backend",
        ml_ready=ml_ready,
        database_ready=db_ready,
        total_students=total_students,
    )


# ── Student Management (Teacher & Admin Only) ────────────────────────────────

@router.get("/students", response_model=List[StudentResponse])
def get_students(
    _: dict = Depends(require_roles("teacher", "admin")),
) -> List[StudentResponse]:
    """
    Return registered student records from the database.
    Restricted to teacher and admin roles.
    """
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT roll_no, name, department, class, registered_at
            FROM students
            ORDER BY name ASC
            """
        )
        rows = cursor.fetchall()
        return [
            StudentResponse(
                roll_no=row["roll_no"],
                name=row["name"],
                department=row["department"],
                class_name=row["class"],
                registered_at=row["registered_at"],
            )
            for row in rows
        ]


# ── Attendance Queries (Teacher & Admin Only) ────────────────────────────────

@router.get("/attendance", response_model=List[AttendanceRecord])
def get_attendance(
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    roll_no: Optional[str] = Query(None, description="Filter by student roll number"),
    _: dict = Depends(require_roles("teacher", "admin")),
) -> List[AttendanceRecord]:
    """
    Return attendance records with optional filtering by date and roll number.
    Restricted to teacher and admin roles.
    """
    query = """
        SELECT a.id, a.roll_no, s.name, s.department, a.date, a.entry_time,
               a.status, a.similarity, a.marked_at
        FROM attendance a
        LEFT JOIN students s ON a.roll_no = s.roll_no
        WHERE 1=1
    """
    params = []
    if date:
        query += " AND a.date = ?"
        params.append(date)
    if roll_no:
        query += " AND a.roll_no = ?"
        params.append(roll_no)

    query += " ORDER BY a.date DESC, a.entry_time DESC"

    with get_db() as conn:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [
            AttendanceRecord(
                id=row["id"],
                roll_no=row["roll_no"],
                name=row["name"],
                department=row["department"],
                date=row["date"],
                entry_time=row["entry_time"],
                status=row["status"],
                similarity=row["similarity"],
                marked_at=row["marked_at"],
            )
            for row in rows
        ]


# ── Student Personal Attendance (Student Only) ───────────────────────────────

@router.get("/attendance/me", response_model=List[AttendanceRecord])
def get_my_attendance(
    current_user: dict = Depends(require_roles("student")),
) -> List[AttendanceRecord]:
    """
    Return attendance records strictly for the authenticated student.
    Derives roll_no from the database user, never from URL query parameters.
    Rejects staff roles (HTTP 403 Forbidden).
    """
    roll_no = current_user.get("roll_no")
    if not roll_no:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Student account is missing a linked roll number.",
        )

    query = """
        SELECT a.id, a.roll_no, s.name, s.department, a.date, a.entry_time,
               a.status, a.similarity, a.marked_at
        FROM attendance a
        LEFT JOIN students s ON a.roll_no = s.roll_no
        WHERE a.roll_no = ?
        ORDER BY a.date DESC, a.entry_time DESC
    """
    with get_db() as conn:
        cursor = conn.execute(query, (roll_no,))
        rows = cursor.fetchall()
        return [
            AttendanceRecord(
                id=row["id"],
                roll_no=row["roll_no"],
                name=row["name"],
                department=row["department"],
                date=row["date"],
                entry_time=row["entry_time"],
                status=row["status"],
                similarity=row["similarity"],
                marked_at=row["marked_at"],
            )
            for row in rows
        ]


# ── Classroom Attendance Excel Export (Teacher & Admin Only) ──────────────────

@router.get("/attendance/export/excel")
def export_attendance_excel(
    report_type: str = Query("monthly", description="Report scope: monthly, semester, or custom"),
    month: Optional[str] = Query(None, description="Month in YYYY-MM format, e.g. 2026-10"),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    current_user: dict = Depends(require_roles("teacher", "admin")),
):
    """
    Generate and download an official attendance report as an Excel (.xlsx) workbook.
    Restricted to teacher and admin roles only.
    Contains:
    - Metadata banner (Generation time, Report period, Generated by)
    - Full list of registered students
    - Columns for each class date on which attendance was conducted
    - P (Present) / A (Absent) indicators with clean cell formatting
    - Per-student totals: Present, Absent, Attendance Percentage
    - Classroom summary statistics
    Uses actual classroom attendance records only (never gate observations or guest visits).
    """
    now = datetime.now()
    if report_type not in ("monthly", "semester", "custom"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported report_type '{report_type}'. Allowed values: monthly, semester, custom.",
        )

    if report_type == "monthly":
        target_month = month or now.strftime("%Y-%m")
        try:
            parts = target_month.split("-")
            if len(parts) != 2:
                raise ValueError()
            year_val, month_val = int(parts[0]), int(parts[1])
            if not (1 <= month_val <= 12 and 1900 <= year_val <= 2100):
                raise ValueError()
            _, last_day = calendar.monthrange(year_val, month_val)
            s_date = f"{year_val:04d}-{month_val:02d}-01"
            e_date = f"{year_val:04d}-{month_val:02d}-{last_day:02d}"
            period_label = f"Month: {datetime(year_val, month_val, 1).strftime('%B %Y')}"
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid month format. Expected YYYY-MM (e.g. 2026-10).",
            )
    elif report_type == "semester":
        # Academic semester: Jul-Dec (Odd) or Jan-Jun (Even) or custom semester dates
        s_date = start_date or (f"{now.year}-07-01" if now.month >= 7 else f"{now.year}-01-01")
        e_date = end_date or (f"{now.year}-12-31" if now.month >= 7 else f"{now.year}-06-30")
        try:
            datetime.strptime(s_date, "%Y-%m-%d")
            datetime.strptime(e_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Expected YYYY-MM-DD.",
            )
        if s_date > e_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date range: Start date cannot be after end date.",
            )
        if not (start_date or end_date):
            period_label = f"Semester: {'July - December' if now.month >= 7 else 'January - June'} {now.year} ({'Odd' if now.month >= 7 else 'Even'} Semester)"
        else:
            period_label = f"Semester Period: {s_date} to {e_date}"
    else:  # custom
        s_date = start_date or f"{now.strftime('%Y-%m')}-01"
        e_date = end_date or now.strftime("%Y-%m-%d")
        try:
            datetime.strptime(s_date, "%Y-%m-%d")
            datetime.strptime(e_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Expected YYYY-MM-DD.",
            )
        if s_date > e_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date range: Start date cannot be after end date.",
            )
        period_label = f"Period: {s_date} to {e_date}"

    with get_db() as conn:
        # 1. Fetch registered students
        students = conn.execute(
            "SELECT roll_no, name, department, class FROM students ORDER BY roll_no ASC"
        ).fetchall()

        # 2. Fetch classroom attendance records strictly from attendance table
        # (Gate observations in security_events are never included)
        att_rows = conn.execute(
            """
            SELECT roll_no, date, status, entry_time
            FROM attendance
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC, entry_time ASC
            """,
            (s_date, e_date),
        ).fetchall()

    # Determine unique class session dates actually conducted
    session_dates = sorted(list(set(r["date"] for r in att_rows)))

    # Map attendance: (roll_no, date) -> True (present)
    present_map = set((r["roll_no"], r["date"]) for r in att_rows if r["status"] == "present")

    # Create Excel workbook
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Classroom Attendance"
    ws.views.sheetView[0].showGridLines = True

    # Styling definitions
    title_font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
    subtitle_font = Font(name="Calibri", size=10, italic=True, color="D9D9D9")
    meta_font = Font(name="Calibri", size=9, bold=True, color="FFFFFF")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10)
    data_bold = Font(name="Calibri", size=10, bold=True)
    pct_font = Font(name="Calibri", size=10, bold=True)
    present_font = Font(name="Calibri", size=10, bold=True, color="276A3C")
    absent_font = Font(name="Calibri", size=10, bold=True, color="A61C1C")

    title_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_fill = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    present_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    absent_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    summary_fill = PatternFill(start_color="EDEDED", end_color="EDEDED", fill_type="solid")
    accent_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    header_border = Border(
        left=Side(style="thin", color="FFFFFF"),
        right=Side(style="thin", color="FFFFFF"),
        top=Side(style="medium", color="1F4E79"),
        bottom=Side(style="medium", color="1F4E79"),
    )

    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")

    total_cols = 4 + len(session_dates) + 4  # Roll, Name, Dept, Class + Dates + Total P, Total A, Total Sessions, %

    # Banner Header Rows
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(total_cols, 8))
    t_cell = ws.cell(row=1, column=1, value="SMARTATTENDANCE — OFFICIAL CLASSROOM ATTENDANCE REGISTER")
    t_cell.font = title_font
    t_cell.fill = title_fill
    t_cell.alignment = align_center

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(total_cols, 8))
    sub_cell = ws.cell(row=2, column=1, value=f"{period_label} | Total Conducted Sessions: {len(session_dates)}")
    sub_cell.font = subtitle_font
    sub_cell.fill = title_fill
    sub_cell.alignment = align_center

    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=max(total_cols, 8))
    m_cell = ws.cell(
        row=3,
        column=1,
        value=f"Generated On: {now.strftime('%Y-%m-%d %H:%M:%S')} | Generated By: {current_user.get('username', 'Teacher')} | Attendance Source: Classroom Attendance Only (Gate Observations Excluded)",
    )
    m_cell.font = meta_font
    m_cell.fill = title_fill
    m_cell.alignment = align_center

    # Row 4 is blank
    # Row 5: Column Headers
    headers = ["Roll No", "Student Name", "Department", "Class"]
    for d in session_dates:
        parts = d.split("-")
        headers.append(f"{parts[1]}/{parts[2]}" if len(parts) == 3 else d)
    headers.extend(["Present", "Absent", "Total Classes Held", "Attendance %"])

    for col_idx, h_text in enumerate(headers, start=1):
        c = ws.cell(row=5, column=col_idx, value=h_text)
        c.font = header_font
        c.fill = header_fill
        c.alignment = align_center
        c.border = header_border

    # Data Rows
    current_row = 6
    total_conducted = len(session_dates)

    for stu in students:
        r_no = stu["roll_no"]
        s_name = stu["name"]
        dept = stu["department"] or "Computer Science"
        cls_name = stu["class"] or "Year 3"

        present_count = 0
        absent_count = 0

        ws.cell(row=current_row, column=1, value=r_no).alignment = align_center
        ws.cell(row=current_row, column=2, value=s_name).alignment = align_left
        ws.cell(row=current_row, column=3, value=dept).alignment = align_center
        ws.cell(row=current_row, column=4, value=cls_name).alignment = align_center

        for col_num in range(1, 5):
            cell = ws.cell(row=current_row, column=col_num)
            cell.font = data_bold if col_num <= 2 else data_font
            cell.border = thin_border

        col_pos = 5
        for d in session_dates:
            is_present = (r_no, d) in present_map
            c = ws.cell(row=current_row, column=col_pos)
            if is_present:
                present_count += 1
                setattr(c, "value", "P")
                c.font = present_font
                c.fill = present_fill
            else:
                absent_count += 1
                setattr(c, "value", "A")
                c.font = absent_font
                c.fill = absent_fill
            c.alignment = align_center
            c.border = thin_border
            col_pos += 1

        pct = round((present_count / total_conducted * 100), 1) if total_conducted > 0 else 0.0

        p_cell = ws.cell(row=current_row, column=col_pos, value=present_count)
        p_cell.font = data_bold
        p_cell.alignment = align_center
        p_cell.border = thin_border
        p_cell.fill = summary_fill

        a_cell = ws.cell(row=current_row, column=col_pos + 1, value=absent_count)
        a_cell.font = data_bold
        a_cell.alignment = align_center
        a_cell.border = thin_border
        a_cell.fill = summary_fill

        t_cell = ws.cell(row=current_row, column=col_pos + 2, value=total_conducted)
        t_cell.font = data_bold
        t_cell.alignment = align_center
        t_cell.border = thin_border
        t_cell.fill = summary_fill

        pct_cell = ws.cell(row=current_row, column=col_pos + 3, value=f"{pct}%")
        pct_cell.font = pct_font
        pct_cell.alignment = align_center
        pct_cell.border = thin_border
        pct_cell.fill = accent_fill

        current_row += 1

    # Adjust column widths automatically
    for col in ws.columns:
        col_idx = col[0].column
        if col_idx is not None:
            col_letter = get_column_letter(col_idx)
            if col_idx == 1:
                ws.column_dimensions[col_letter].width = 14
            elif col_idx == 2:
                ws.column_dimensions[col_letter].width = 24
            elif col_idx in (3, 4):
                ws.column_dimensions[col_letter].width = 16
            elif col_idx > total_cols - 4:
                ws.column_dimensions[col_letter].width = 14
            else:
                ws.column_dimensions[col_letter].width = 10

    # Save to buffer
    output_buf = io.BytesIO()
    wb.save(output_buf)
    output_buf.seek(0)

    filename = f"SmartAttendance_{report_type}_{s_date}_to_{e_date}.xlsx"
    return Response(
        content=output_buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# ── Classroom Attendance Marking (Teacher & Admin Only) ──────────────────────
# Security observations at the gate must NEVER mark classroom attendance.

@router.post("/attendance/mark")
def mark_attendance(
    payload: MarkAttendanceRequest,
    _: dict = Depends(require_roles("teacher", "admin")),
):
    """
    Mark classroom attendance for a validated student.
    Strictly restricted to authorized teacher and admin roles.

    Security personnel and gate monitoring cameras must NOT call this endpoint.
    Gate observations are recorded separately in security_events.

    Rules:
    - Verifies identity exists in the students table.
    - Prevents duplicate attendance for the same student on the same date (HTTP 409).
    """
    with get_db() as conn:
        student = None
        if payload.roll_no:
            cursor = conn.execute(
                "SELECT roll_no, name FROM students WHERE roll_no = ?",
                (payload.roll_no,),
            )
            student = cursor.fetchone()

        if not student and payload.name:
            cursor = conn.execute(
                "SELECT roll_no, name FROM students WHERE LOWER(name) = LOWER(?)",
                (payload.name,),
            )
            student = cursor.fetchone()

        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found in registered database. Attendance rejected.",
            )

        roll_no = student["roll_no"]
        student_name = student["name"]
        today = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().strftime("%H:%M:%S")

        # Check for duplicate
        cursor = conn.execute(
            "SELECT id, entry_time FROM attendance WHERE roll_no = ? AND date = ?",
            (roll_no, today),
        )
        existing = cursor.fetchone()
        if existing:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "status": "duplicate",
                    "message": f"Attendance already recorded for {student_name} today at {existing['entry_time']}.",
                    "roll_no": roll_no,
                    "date": today,
                    "entry_time": existing["entry_time"],
                },
            )

        # Insert attendance entry
        conn.execute(
            """
            INSERT INTO attendance (roll_no, date, entry_time, status, similarity)
            VALUES (?, ?, ?, 'present', ?)
            """,
            (roll_no, today, now_time, payload.similarity),
        )

        return {
            "status": "success",
            "message": f"Attendance marked for {student_name}.",
            "roll_no": roll_no,
            "name": student_name,
            "date": today,
            "entry_time": now_time,
            "similarity": payload.similarity,
        }


# ── Helper: Active Guests ─────────────────────────────────────────────────────

def _get_active_guests(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all approved guests whose validity period covers the current timestamp."""
    now_iso = datetime.now().isoformat()
    cursor = conn.execute(
        """
        SELECT guest_id, full_name, valid_until, face_embedding
        FROM guests
        WHERE status = 'approved'
          AND valid_from <= ?
          AND valid_until >= ?
        """,
        (now_iso, now_iso),
    )
    return [dict(row) for row in cursor.fetchall()]


# ── Face Recognition (Security, Teacher & Admin) ──────────────────────────────

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Query(None, description="Camera / session identifier for event deduplication"),
    current_user: dict = Depends(require_roles("security", "teacher", "admin")),
):
    """
    Accept an uploaded webcam frame and run multi-identity recognition against
    registered students and active approved guests.
    Restricted to security, teacher, and admin roles.

    Every frame is analysed and live bounding-box data returned to the caller.
    Persistent database writes (recognition_log, security_events) happen ONLY
    on meaningful transitions:
      - First confirmed sighting of a new visit  → recognition_log row
      - First confirmed UNKNOWN sighting          → security_events UNKNOWN_ALERT
      - First confirmed GATE sighting             → security_events GATE_OBSERVATION
    Repeated frames of the same continuously-visible identity do NOT create
    additional rows.
    """
    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type: {file.content_type}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image size exceeds maximum allowed size (10MB).",
        )

    ml_service = MLService.get_instance()
    if not ml_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition service is currently unavailable.",
        )

    with get_db() as conn:
        active_guests = _get_active_guests(conn)

    try:
        results = ml_service.recognize_bytes(contents, active_guests=active_guests)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception("Error during face recognition: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing face recognition.",
        )

    # Determine camera identifier for per-camera tracking
    cam_id = (camera_id or current_user.get("username", "default")).strip() or "default"
    is_gate_cam = (cam_id == "gate_webcam" or "gate" in cam_id.lower() or current_user.get("role") == "security")
    tracker = get_tracker()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db() as conn:
        for r in results:
            category = r["identity_category"]
            name = r["name"]
            status_str = r["status"]
            guest_id = r.get("guest_id")

            # NO_FACE frames: still update bounding box on frontend but skip all DB writes
            if status_str == "NO_FACE":
                continue

            # Consult the in-memory tracker to decide whether a DB write is needed
            should_write_log, should_write_event, _ = tracker.observe(
                identity_category=category,
                name=name,
                camera_id=cam_id,
                guest_id=guest_id,
                is_gate_observation=is_gate_cam,
            )

            # 1. recognition_log — only on first confirmed sighting of each new visit
            if should_write_log:
                conn.execute(
                    """
                    INSERT INTO recognition_log (
                        name, similarity, status, bbox_x, bbox_y, bbox_w, bbox_h,
                        detection_score, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'api')
                    """,
                    (
                        name,
                        r["similarity"],
                        status_str,
                        r["bbox"]["x"],
                        r["bbox"]["y"],
                        r["bbox"]["w"],
                        r["bbox"]["h"],
                        r["detection_score"],
                    ),
                )
                logger.debug("recognition_log: new visit for %s on cam=%s", name, cam_id)

            # 2. security_events: UNKNOWN_ALERT vs GATE_OBSERVATION
            if category == "UNKNOWN_PERSON":
                if should_write_event:
                    evt_id = f"EVT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
                    track_key_val = f"UNKNOWN:{cam_id}"
                    conn.execute(
                        """
                        INSERT INTO security_events (
                            event_id, timestamp, event_type, identity_category, name,
                            similarity, source_camera, bbox_x, bbox_y, bbox_w, bbox_h,
                            resolution_status, track_key, last_seen_at
                        ) VALUES (?, ?, 'UNKNOWN_ALERT', 'UNKNOWN', 'UNKNOWN PERSON', ?,
                                  ?, ?, ?, ?, ?, 'unresolved', ?, ?)
                        """,
                        (
                            evt_id,
                            now_str,
                            r["similarity"],
                            cam_id,
                            r["bbox"]["x"],
                            r["bbox"]["y"],
                            r["bbox"]["w"],
                            r["bbox"]["h"],
                            track_key_val,
                            now_str,
                        ),
                    )
                    tracker.set_event_id(
                        identity_category=category,
                        name=name,
                        camera_id=cam_id,
                        event_id=evt_id,
                        guest_id=guest_id,
                    )
                    logger.info("security_events: UNKNOWN_ALERT %s on cam=%s", evt_id, cam_id)

                elif not should_write_event:
                    # Active track — update last_seen_at in-place (no new row)
                    conn.execute(
                        """
                        UPDATE security_events
                        SET last_seen_at = ?
                        WHERE track_key = ? AND resolution_status = 'unresolved'
                          AND id = (SELECT MAX(id) FROM security_events
                                    WHERE track_key = ? AND resolution_status = 'unresolved')
                        """,
                        (now_str, f"UNKNOWN:{cam_id}", f"UNKNOWN:{cam_id}"),
                    )

            elif is_gate_cam and category in ("REGISTERED_STUDENT", "APPROVED_GUEST"):
                # Distinct gate entry observation (NOT classroom attendance)
                track_key_val = (
                    f"STUDENT:{name}:{cam_id}"
                    if category == "REGISTERED_STUDENT"
                    else f"GUEST:{guest_id or name}:{cam_id}"
                )
                identity_cat_label = "STUDENT" if category == "REGISTERED_STUDENT" else "GUEST"

                if should_write_event:
                    evt_id = f"EVT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
                    conn.execute(
                        """
                        INSERT INTO security_events (
                            event_id, timestamp, event_type, identity_category, name,
                            similarity, source_camera, bbox_x, bbox_y, bbox_w, bbox_h,
                            resolution_status, track_key, last_seen_at
                        ) VALUES (?, ?, 'GATE_OBSERVATION', ?, ?, ?,
                                  ?, ?, ?, ?, ?, 'resolved', ?, ?)
                        """,
                        (
                            evt_id,
                            now_str,
                            identity_cat_label,
                            name,
                            r["similarity"],
                            cam_id,
                            r["bbox"]["x"],
                            r["bbox"]["y"],
                            r["bbox"]["w"],
                            r["bbox"]["h"],
                            track_key_val,
                            now_str,
                        ),
                    )
                    tracker.set_event_id(
                        identity_category=category,
                        name=name,
                        camera_id=cam_id,
                        event_id=evt_id,
                        guest_id=guest_id,
                    )
                    logger.info("security_events: GATE_OBSERVATION %s for %s on cam=%s", evt_id, name, cam_id)

                elif not should_write_event:
                    # Active gate observation track — update last_seen_at in-place
                    conn.execute(
                        """
                        UPDATE security_events
                        SET last_seen_at = ?
                        WHERE track_key = ?
                          AND id = (SELECT MAX(id) FROM security_events WHERE track_key = ?)
                        """,
                        (now_str, track_key_val, track_key_val),
                    )

    face_results = [FaceRecognitionResult(**r) for r in results]
    return RecognitionResponse(
        faces_detected=len([r for r in results if r["status"] != "NO_FACE"]),
        results=face_results,
    )


# ── Guest Management Endpoints ────────────────────────────────────────────────

@router.post("/guests/register", response_model=GuestResponse)
def register_guest(
    payload: GuestRegisterRequest,
    current_user: dict = Depends(require_roles("security", "admin")),
):
    """
    Register and approve a guest visitor at the security gate.
    Requires an authenticated security or admin officer.

    - Validates guest details.
    - Generates 128-d facial embedding from the provided face sample.
    - Ensures exactly 1 face exists in the registration sample.
    - Saves guest into database with approved status and validity period.
    - Creates a security audit event.
    """
    if not payload.full_name or not payload.full_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Guest full name is required.",
        )
    if not payload.purpose or not payload.purpose.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Purpose of visit is required.",
        )
    if not payload.host_to_meet or not payload.host_to_meet.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Host / person to meet is required.",
        )
    if not payload.image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Guest facial image data (image_base64) is required.",
        )

    # Decode base64 image data
    img_data_str = payload.image_base64
    if "," in img_data_str:
        img_data_str = img_data_str.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(img_data_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 encoding for guest facial image.",
        )

    ml_service = MLService.get_instance()
    if not ml_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition service is not ready.",
        )

    try:
        emb, thumb_b64, bbox_dict = ml_service.extract_guest_face_template(image_bytes)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )

    # Check validity window
    now_dt = datetime.now()
    valid_from = payload.valid_from or now_dt.strftime("%Y-%m-%dT%H:%M:%S")
    valid_until = payload.valid_until or f"{now_dt.strftime('%Y-%m-%d')}T23:59:59"

    # Generate unique guest ID
    guest_id = f"GST-{now_dt.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    created_at = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    approved_by = current_user.get("username", "security")

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO guests (
                guest_id, full_name, purpose, host_to_meet, phone,
                valid_from, valid_until, status, face_embedding,
                face_snapshot, approved_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'approved', ?, ?, ?, ?)
            """,
            (
                guest_id,
                payload.full_name.strip(),
                payload.purpose.strip(),
                payload.host_to_meet.strip(),
                (payload.phone or "").strip(),
                valid_from,
                valid_until,
                emb.tobytes(),
                thumb_b64,
                approved_by,
                created_at,
            ),
        )

        # Log security event for audit trail
        evt_id = f"EVT-{now_dt.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        conn.execute(
            """
            INSERT INTO security_events (
                event_id, timestamp, event_type, identity_category, name,
                similarity, source_camera, bbox_x, bbox_y, bbox_w, bbox_h,
                snapshot, resolution_status, resolved_by, resolved_at, resolution_notes
            ) VALUES (?, ?, 'GUEST_REGISTERED', 'GUEST', ?, 1.0, 'gate_webcam', ?, ?, ?, ?, ?, 'registered_as_guest', ?, ?, ?)
            """,
            (
                evt_id,
                created_at,
                payload.full_name.strip(),
                bbox_dict["x"],
                bbox_dict["y"],
                bbox_dict["w"],
                bbox_dict["h"],
                thumb_b64,
                approved_by,
                created_at,
                f"Guest approved: {guest_id}. Purpose: {payload.purpose.strip()}",
            ),
        )

    return GuestResponse(
        guest_id=guest_id,
        full_name=payload.full_name.strip(),
        purpose=payload.purpose.strip(),
        host_to_meet=payload.host_to_meet.strip(),
        phone=(payload.phone or "").strip(),
        valid_from=valid_from,
        valid_until=valid_until,
        status="approved",
        face_snapshot=thumb_b64,
        approved_by=approved_by,
        created_at=created_at,
    )


@router.post("/guests/register-upload", response_model=GuestResponse)
async def register_guest_upload(
    full_name: str = Form(...),
    purpose: str = Form(...),
    host_to_meet: str = Form(...),
    phone: Optional[str] = Form(""),
    valid_from: Optional[str] = Form(None),
    valid_until: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: dict = Depends(require_roles("security", "admin")),
):
    """Multipart form endpoint for registering a guest via direct file upload."""
    contents = await file.read()
    b64_str = base64.b64encode(contents).decode("utf-8")
    req = GuestRegisterRequest(
        full_name=full_name,
        purpose=purpose,
        host_to_meet=host_to_meet,
        phone=phone or "",
        valid_from=valid_from,
        valid_until=valid_until,
        image_base64=b64_str,
    )
    return register_guest(req, current_user=current_user)


@router.get("/guests", response_model=List[GuestResponse])
def get_guests(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status: approved, revoked, all"),
    _: dict = Depends(require_roles("security", "teacher", "admin")),
) -> List[GuestResponse]:
    """
    List registered guests.
    Accessible to security, teacher, and admin roles.
    Never exposes raw biometric embeddings.
    """
    query = """
        SELECT guest_id, full_name, purpose, host_to_meet, phone,
               valid_from, valid_until, status, face_snapshot,
               approved_by, created_at, revoked_by, revoked_at, revocation_reason
        FROM guests
    """
    params = []
    if status_filter and status_filter.lower() != "all":
        query += " WHERE status = ?"
        params.append(status_filter.lower())

    query += " ORDER BY created_at DESC"

    with get_db() as conn:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

    return [
        GuestResponse(
            guest_id=r["guest_id"],
            full_name=r["full_name"],
            purpose=r["purpose"],
            host_to_meet=r["host_to_meet"],
            phone=r["phone"],
            valid_from=r["valid_from"],
            valid_until=r["valid_until"],
            status=r["status"],
            face_snapshot=r["face_snapshot"],
            approved_by=r["approved_by"],
            created_at=r["created_at"],
            revoked_by=r["revoked_by"],
            revoked_at=r["revoked_at"],
            revocation_reason=r["revocation_reason"],
        )
        for r in rows
    ]


@router.get("/guests/{guest_id}", response_model=GuestResponse)
def get_guest_by_id(
    guest_id: str,
    _: dict = Depends(require_roles("security", "teacher", "admin")),
) -> GuestResponse:
    """Retrieve details of a specific guest by ID."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT guest_id, full_name, purpose, host_to_meet, phone,
                   valid_from, valid_until, status, face_snapshot,
                   approved_by, created_at, revoked_by, revoked_at, revocation_reason
            FROM guests
            WHERE guest_id = ?
            """,
            (guest_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Guest with ID '{guest_id}' not found.",
            )

        return GuestResponse(
            guest_id=row["guest_id"],
            full_name=row["full_name"],
            purpose=row["purpose"],
            host_to_meet=row["host_to_meet"],
            phone=row["phone"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            status=row["status"],
            face_snapshot=row["face_snapshot"],
            approved_by=row["approved_by"],
            created_at=row["created_at"],
            revoked_by=row["revoked_by"],
            revoked_at=row["revoked_at"],
            revocation_reason=row["revocation_reason"],
        )


@router.post("/guests/{guest_id}/revoke", response_model=GuestResponse)
def revoke_guest(
    guest_id: str,
    payload: Optional[GuestRevokeRequest] = None,
    current_user: dict = Depends(require_roles("security", "admin")),
) -> GuestResponse:
    """
    Revoke authorization for a guest.
    Immediately invalidates subsequent recognition matches across all dashboards.
    Requires security or admin role.
    """
    reason = payload.reason if payload else "Revoked by security officer"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = current_user.get("username", "security")

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, full_name, status FROM guests WHERE guest_id = ?",
            (guest_id,),
        )
        guest = cursor.fetchone()
        if not guest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Guest '{guest_id}' does not exist.",
            )

        conn.execute(
            """
            UPDATE guests
            SET status = 'revoked',
                revoked_by = ?,
                revoked_at = ?,
                revocation_reason = ?
            WHERE guest_id = ?
            """,
            (username, now_str, reason, guest_id),
        )

        # Record security audit event
        evt_id = f"EVT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        conn.execute(
            """
            INSERT INTO security_events (
                event_id, timestamp, event_type, identity_category, name,
                similarity, source_camera, resolution_status, resolved_by,
                resolved_at, resolution_notes
            ) VALUES (?, ?, 'GUEST_REVOKED', 'GUEST', ?, 0.0, 'gate_webcam', 'resolved', ?, ?, ?)
            """,
            (
                evt_id,
                now_str,
                guest["full_name"],
                username,
                now_str,
                f"Guest pass revoked: {guest_id}. Reason: {reason}",
            ),
        )

    return get_guest_by_id(guest_id, _=current_user)


# ── Security Event & Unknown Person Alerts ────────────────────────────────────

@router.get("/security/events", response_model=List[SecurityEventResponse])
def get_security_events(
    limit: int = Query(50, ge=1, le=200),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    unresolved_only: bool = Query(False, description="Filter only unresolved events"),
    _: dict = Depends(require_roles("security", "teacher", "admin")),
) -> List[SecurityEventResponse]:
    """
    Retrieve security alerts and events for live dashboard feeds.
    Accessible to security, teacher, and admin roles.
    """
    query = """
        SELECT event_id, timestamp, event_type, identity_category, name,
               similarity, source_camera, bbox_x, bbox_y, bbox_w, bbox_h,
               snapshot, resolution_status, resolved_by, resolved_at, resolution_notes
        FROM security_events
        WHERE 1=1
    """
    params: List[Any] = []
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)
    if unresolved_only:
        query += " AND resolution_status = 'unresolved'"

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

    result = []
    for r in rows:
        bbox = None
        if r["bbox_w"] and r["bbox_h"]:
            bbox = BoundingBox(x=r["bbox_x"] or 0, y=r["bbox_y"] or 0, w=r["bbox_w"], h=r["bbox_h"])
        result.append(
            SecurityEventResponse(
                event_id=r["event_id"],
                timestamp=r["timestamp"],
                event_type=r["event_type"],
                identity_category=r["identity_category"],
                name=r["name"],
                similarity=r["similarity"],
                source_camera=r["source_camera"],
                bbox=bbox,
                snapshot=r["snapshot"],
                resolution_status=r["resolution_status"],
                resolved_by=r["resolved_by"],
                resolved_at=r["resolved_at"],
                resolution_notes=r["resolution_notes"],
            )
        )
    return result


@router.post("/security/events/{event_id}/resolve")
def resolve_security_event(
    event_id: str,
    payload: EventResolveRequest,
    current_user: dict = Depends(require_roles("security", "admin")),
):
    """
    Mark an unknown-person security alert as resolved, dismissed, or investigated.
    Requires security or admin role.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = current_user.get("username", "security")

    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id FROM security_events WHERE event_id = ?",
            (event_id,),
        )
        if not cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Security event '{event_id}' not found.",
            )

        conn.execute(
            """
            UPDATE security_events
            SET resolution_status = ?,
                resolved_by = ?,
                resolved_at = ?,
                resolution_notes = ?
            WHERE event_id = ?
            """,
            (
                payload.resolution_status,
                username,
                now_str,
                payload.resolution_notes or f"Resolved by {username}",
                event_id,
            ),
        )

    return {
        "status": "success",
        "event_id": event_id,
        "resolution_status": payload.resolution_status,
        "resolved_by": username,
        "resolved_at": now_str,
    }


@router.get("/security/stream")
async def security_events_stream(
    current_user: dict = Depends(require_roles("security", "teacher", "admin")),
):
    """
    Server-Sent Events (SSE) stream for real-time security events & alerts.
    Both Security and Teacher dashboards subscribe to this stream for instant sync.
    """
    async def event_generator():
        last_seen_id = 0
        # Initialize last_seen_id with the highest current event ID
        with get_db() as conn:
            cursor = conn.execute("SELECT MAX(id) FROM security_events")
            max_id = cursor.fetchone()[0]
            if max_id:
                last_seen_id = max_id

        try:
            while True:
                with get_db() as conn:
                    cursor = conn.execute(
                        """
                        SELECT id, event_id, timestamp, event_type, identity_category,
                               name, similarity, source_camera, resolution_status
                        FROM security_events
                        WHERE id > ?
                        ORDER BY id ASC
                        """,
                        (last_seen_id,),
                    )
                    rows = cursor.fetchall()

                for row in rows:
                    last_seen_id = row["id"]
                    event_data = {
                        "event_id": row["event_id"],
                        "timestamp": row["timestamp"],
                        "event_type": row["event_type"],
                        "identity_category": row["identity_category"],
                        "name": row["name"],
                        "similarity": row["similarity"],
                        "source_camera": row["source_camera"],
                        "resolution_status": row["resolution_status"],
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"

                # Send heartbeat every iteration
                yield ": heartbeat\n\n"
                await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

