# SmartAttendance 🎓

**AI-Based Face Recognition Attendance Management System using a Laptop Webcam**

> A college project by Prapti Chavan — Vidyalankar Dnyanapeeth Trust  
> Branch: [`Prapti`](https://github.com/praptichavan25/SmartAttendance/tree/Prapti)

---

## Problem Statement

Traditional paper-based and manual attendance systems are slow, error-prone, and easy to misuse through proxy attendance. This project replaces the manual process with an automated, AI-powered system that uses a standard laptop webcam to recognise students by face, log security-gate observations in real time, and let teachers record accurate classroom attendance — all without specialised hardware.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Face Detection** | YuNet (OpenCV) detects faces in every webcam frame with bounding boxes |
| 🧠 **Face Recognition** | SFace generates 128-d embeddings; cosine similarity matches against registered students |
| 🔐 **Role-Based Access** | Four roles: Student, Teacher, Security, Admin — each with its own dashboard and API permissions |
| 📋 **Classroom Attendance** | Teachers mark verified attendance; gate observations never auto-create attendance records |
| 🚪 **Security Gate** | Real-time webcam feed at entry gate logs observations and alerts for unknown persons |
| 👥 **Guest Management** | Security registers authorised guests with face photo; guests shown as orange box on scan |
| 📊 **Excel Reports** | Teachers download monthly, semester, or custom date-range attendance reports (`.xlsx`) |
| 🔄 **Deduplication** | Repeated frames of the same face within 30 s produce only one event; 60 s cooldown before re-logging |
| 🧪 **75 Automated Tests** | Full test coverage: auth, RBAC, recognition, deduplication, Excel export, guest security |

---

## Technology Stack

### Backend
| Component | Technology |
|-----------|-----------|
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) 0.141 |
| ASGI Server | [Uvicorn](https://www.uvicorn.org/) 0.52 |
| Database | SQLite 3 (via Python `sqlite3`) |
| Authentication | [PyJWT](https://pyjwt.readthedocs.io/) 2.15 + [bcrypt](https://pypi.org/project/bcrypt/) 5.0 |
| Data Validation | [Pydantic](https://docs.pydantic.dev/) 2.12 |
| Excel Export | [openpyxl](https://openpyxl.readthedocs.io/) 3.1 |
| Config | [python-dotenv](https://pypi.org/project/python-dotenv/) |

### ML / Face Recognition
| Component | Technology |
|-----------|-----------|
| Face Detection | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (ONNX, via `cv2.FaceDetectorYN`) |
| Face Recognition | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (ONNX, via `cv2.FaceRecognizerSF`) |
| Computer Vision | [OpenCV](https://opencv.org/) 4.14 |
| Numerical | [NumPy](https://numpy.org/) 2.5 |
| Python | 3.13+ |

### Frontend
- Vanilla HTML5, CSS3, JavaScript (no framework)
- Shared design system (`shared.css`)
- `auth.js` — centralised JWT session management

### Testing
| Tool | Purpose |
|------|---------|
| pytest 9.1 | Test runner |
| FastAPI TestClient | In-process HTTP API testing |
| httpx 0.28 | HTTP transport for TestClient |

---

## System Architecture

```
SmartAttendance/
├── backend/                    # FastAPI REST API
│   ├── app/
│   │   ├── main.py             # FastAPI application factory + lifespan
│   │   ├── config.py           # Environment-driven configuration
│   │   ├── auth.py             # bcrypt hashing, JWT tokens, user management
│   │   ├── recognition_tracker.py  # Frame deduplication (active tracks + cooldown)
│   │   ├── api/
│   │   │   ├── endpoints.py    # All API route handlers (1 400 lines)
│   │   │   ├── schemas.py      # Pydantic request/response models
│   │   │   └── dependencies.py # JWT auth middleware, role enforcement
│   │   ├── db/
│   │   │   └── database.py     # SQLite schema, init_db(), seeding
│   │   └── ml/
│   │       └── service.py      # MLService singleton — bridges FastAPI ↔ ml/src
│   ├── tests/                  # 75 automated tests (7 test files)
│   ├── requirements.txt
│   ├── run.py                  # Convenience runner
│   └── .env.example            # Environment variable template
│
├── ml/                         # Face recognition module (standalone Python)
│   ├── src/
│   │   ├── face_detector.py    # YuNetDetector wrapper
│   │   ├── face_recognizer.py  # FaceRecognizer: detect → embed → match
│   │   ├── generate_embeddings.py  # Offline: build embeddings.pkl from photos
│   │   └── live_recognition.py # Live webcam demo (standalone, no server needed)
│   ├── models/                 # ONNX model files (not committed — see Setup)
│   │   ├── face_detection_yunet_2023mar.onnx   (227 KB)
│   │   └── face_recognition_sface_2021dec.onnx (37 MB)
│   ├── dataset/
│   │   └── registered/         # Per-person photo folders (NOT committed — private)
│   ├── embeddings/             # embeddings.pkl (NOT committed — contains biometric data)
│   └── requirements.txt        # numpy, opencv-python
│
├── frontend/                   # Web dashboards (served as static files)
│   ├── index.html              # Login page (role selector + credential form)
│   ├── student_dashboard.html  # Student: view own attendance
│   ├── teacher_dashboard.html  # Teacher: roster, mark attendance, Excel download
│   ├── security_dashboard.html # Security: webcam recognition, guest management
│   ├── auth.js                 # JWT session helper (shared by all dashboards)
│   └── shared.css              # Common design tokens and layout
│
├── frontend_reference/         # Original reference HTML files (preserved, unchanged)
├── pyrefly.toml                # Pyrefly type-checker config (IDE use)
├── pyrightconfig.json          # Pyright type-checker config (IDE use)
└── .gitignore
```

---

## How Face Detection & Recognition Work

### Detection — YuNet
Each webcam frame is passed to `cv2.FaceDetectorYN` (YuNet ONNX model). YuNet detects faces as bounding boxes of shape `[x, y, w, h, lm0_x, lm0_y, …, score]` — a lightweight, real-time detector that works well on a laptop CPU.

### Recognition — SFace
For each detected face:
1. `cv2.FaceRecognizerSF.alignCrop()` crops and aligns the face to a canonical 112×112 patch.
2. `cv2.FaceRecognizerSF.feature()` produces a **128-dimensional L2-normalised embedding vector**.
3. The embedding is compared against every stored embedding in `embeddings.pkl` using **cosine similarity** (range 0–1, higher = more similar).
4. If the best match exceeds the threshold (`COSINE_THRESHOLD = 0.38`, configurable), the face is recognised as that person; otherwise it is classified as **UNKNOWN**.

### Classification
| Similarity vs Threshold | Result | Box Colour |
|------------------------|--------|-----------|
| ≥ threshold → student match | Registered Student | 🟩 Green |
| ≥ threshold → guest match | Approved Guest | 🟠 Orange |
| Below threshold, clear face | Unknown Person | 🟥 Red |
| Low detection confidence / tiny face | Inconclusive | 🟨 Yellow |

### Deduplication
`RecognitionTracker` prevents flooding the database with duplicate events:
- A continuously-visible face is treated as one **active track** for **30 seconds**.
- After the person leaves frame, a **60-second cooldown** must elapse before the same identity produces a new event.
- At least **2 consecutive frames** must confirm the same identity before any event is persisted.

---

## Student Registration Workflow

1. Collect **multiple photos** of each student in different lighting/angles.
2. Place photos in `ml/dataset/registered/<StudentName>/photo1.jpg` etc.
3. Run the embedding generator:
   ```bash
   cd ml
   .venv/bin/python src/generate_embeddings.py
   # → writes ml/embeddings/embeddings.pkl
   ```
4. Restart the backend — `init_db()` seeds the `students` table from `embeddings.pkl`.

---

## Gate Observations vs Classroom Attendance

This distinction is a core design decision of the system:

| Aspect | Security Gate | Classroom Attendance |
|--------|--------------|----------------------|
| **Who creates it** | Automatic — from webcam recognition | Manual — teacher clicks "Mark Present" |
| **Where stored** | `security_events` table | `attendance` table |
| **Appears in Excel** | ❌ Never | ✅ Always |
| **RBAC** | Security role | Teacher / Admin role |
| **Purpose** | Campus access logging | Academic attendance record |

A student walking past the gate camera **never** creates a classroom attendance record. Classroom attendance is only created when a teacher explicitly marks it via the Teacher Dashboard or the `POST /api/attendance/mark` API.

---

## Authentication & Roles

JWT-based authentication. Tokens stored in `sessionStorage` (never `localStorage` or URL).

| Role | Login | Permissions |
|------|-------|-------------|
| **student** | Username + password | View own attendance (`GET /api/attendance/me`) |
| **teacher** | Username + password | View all students, view/mark attendance, download Excel |
| **security** | Username + password | Face recognition, guest registration/revocation, security events |
| **admin** | Username + password | Everything teacher + security can do |

Protected routes return `403 Forbidden` for the wrong role and `401 Unauthorized` for expired/missing tokens.

---

## API Endpoints

| Method | Path | Roles | Description |
|--------|------|-------|-------------|
| `POST` | `/api/auth/login` | Public | Authenticate; receive JWT |
| `GET` | `/api/auth/me` | Any authenticated | Return safe user profile |
| `GET` | `/api/health` | Public | System status + ML readiness |
| `GET` | `/api/students` | teacher, admin | Full student roster |
| `GET` | `/api/attendance` | teacher, admin | All attendance records |
| `GET` | `/api/attendance/me` | student | Own attendance records |
| `POST` | `/api/attendance/mark` | teacher, admin | Mark a student present/absent |
| `GET` | `/api/attendance/export/excel` | teacher, admin | Download `.xlsx` report |
| `POST` | `/api/recognize` | security, admin | Recognise a face from uploaded image |
| `POST` | `/api/guests/register` | security, admin | Register a guest with face photo |
| `GET` | `/api/guests` | security, admin | List active guests |
| `POST` | `/api/guests/{id}/revoke` | security, admin | Revoke guest access |
| `GET` | `/api/security/events` | security, admin | List gate security events |
| `POST` | `/api/security/events/{id}/resolve` | security, admin | Resolve a security event |
| `GET` | `/api/security/events/stream` | security, admin | Server-Sent Events stream |

Interactive API docs: `http://127.0.0.1:8000/docs`

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| macOS | Ventura 13+ (tested) |
| Python | 3.13+ |
| pip | bundled with Python |
| Git | any recent version |
| Browser | Chrome, Edge, or Firefox (for webcam access) |
| Webcam | Built-in laptop webcam or USB webcam |

> The YuNet detector model (~227 KB) is auto-downloaded on first run if missing.  
> The SFace recogniser model (~37 MB) must be placed manually — see Setup Step 4.

---

## Installation & Setup (macOS)

### 1. Clone the repository

```bash
git clone https://github.com/praptichavan25/SmartAttendance.git
cd SmartAttendance
git checkout Prapti
```

### 2. Create the backend virtual environment

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
cd ..
```

### 3. Create the ML virtual environment

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
cd ..
```

### 4. Obtain the ML model files

The ONNX model files are not committed to Git (the SFace model is 37 MB).

**Option A — Auto-download (YuNet only):** The first time you run the backend, `face_detector.py` will automatically download the YuNet model from the OpenCV Zoo.

**Option B — Manual download (recommended for both models):**

```bash
mkdir -p ml/models
# YuNet (227 KB)
curl -L "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
     -o ml/models/face_detection_yunet_2023mar.onnx

# SFace (37 MB)
curl -L "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
     -o ml/models/face_recognition_sface_2021dec.onnx
```

### 5. Configure environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env:
#   ENVIRONMENT=development
#   JWT_SECRET_KEY=<any-string-for-development>
```

### 6. Register student faces & generate embeddings

```bash
# Create one sub-folder per student under ml/dataset/registered/
mkdir -p ml/dataset/registered/Alice
# Copy several clear facial photos into each folder
# Then generate embeddings:
cd ml
.venv/bin/python src/generate_embeddings.py
cd ..
```

### 7. Initialise the database (automatic on first start)

The database and all tables are created automatically when the backend first starts.

---

## Running the Application

Open **two separate terminal windows** from the project root.

### Terminal 1 — Backend

```bash
cd backend
.venv/bin/python run.py
```

Expected output:
```
Starting SmartAttendance Backend on http://127.0.0.1:8000...
Initializing database...
Pre-warming ML models...
FaceRecognizer ready — 9 registered people, 20 embedding(s) loaded. Threshold: 0.38
Backend application startup complete.
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Terminal 2 — Frontend

```bash
cd frontend
python3 -m http.server 8080
```

### Open in browser

| URL | Page |
|-----|------|
| `http://localhost:8080/` | Login |
| `http://localhost:8080/teacher_dashboard.html` | Teacher Dashboard |
| `http://localhost:8080/security_dashboard.html` | Security Dashboard |
| `http://localhost:8080/student_dashboard.html` | Student Dashboard |
| `http://127.0.0.1:8000/docs` | API Documentation (Swagger UI) |
| `http://127.0.0.1:8000/api/health` | Health check |

> ⚠️ **Important:** Open the frontend at `http://localhost:8080` — not `file://`. The browser's camera API and CORS require an HTTP origin.

---

## Development Accounts

These accounts are provisioned automatically in **development mode** (`ENVIRONMENT=development`):

| Username | Password | Role |
|----------|----------|------|
| `teacher` | `Teacher@123` | teacher |
| `admin` | `Admin@123` | admin |
| `security` | `Security@123` | security |
| *(student names)* | `Student@123` | student |

> 🔒 **Change all passwords before any production deployment.**

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure:

```env
# Server
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

# CORS — only the local frontend origin is permitted
CORS_ORIGIN=http://localhost:8080

# Database (relative to backend/)
DATABASE_URL=sqlite:///./smartattendance.db

# ML model paths (leave blank for auto-detection relative to ml/)
ML_EMBEDDINGS_PATH=
ML_YUNET_MODEL_PATH=
ML_SFACE_MODEL_PATH=

# Recognition threshold — do not lower without testing
COSINE_THRESHOLD=0.38

# Auth
ENVIRONMENT=development
JWT_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

> `JWT_SECRET_KEY` **must** be set to a strong random value in production. The app raises a `RuntimeError` on startup if `ENVIRONMENT=production` and the key is missing.

---

## Running Tests

```bash
# From the project root
backend/.venv/bin/python -m pytest backend/tests/ -v
```

Expected result:
```
75 passed in ~7s
```

Run a specific file:
```bash
backend/.venv/bin/python -m pytest backend/tests/test_excel_export.py -v
```

### Test coverage

| File | What it tests |
|------|--------------|
| `test_auth_foundation.py` | Password hashing, JWT creation/decoding, token expiry |
| `test_auth_rbac.py` | Login flows, protected endpoint RBAC for all 4 roles |
| `test_backend.py` | Health, students, attendance mark/duplicate, recognition API |
| `test_deduplication.py` | RecognitionTracker confirm frames, cooldowns, camera isolation |
| `test_excel_export.py` | Monthly/semester/custom Excel generation, RBAC, edge cases |
| `test_gate_vs_classroom.py` | Gate ≠ classroom, deduplication, RBAC, Excel exclusion proof |
| `test_guest_security.py` | Guest register, recognize, revoke, expiry, embedding privacy |

---

## Troubleshooting

### Backend: `Address already in use`
```bash
# Find and kill the process on port 8000
lsof -ti :8000 | xargs kill -9
```

### Frontend: Port 8080 already in use
```bash
lsof -ti :8080 | xargs kill -9
```

### Camera: "Camera not working" or permission denied
- Open the frontend at `http://localhost:8080` (not `127.0.0.1:8080` or `file://`).
- In Chrome: Settings → Privacy → Camera → allow `localhost`.
- macOS: System Settings → Privacy & Security → Camera → enable your browser.

### Backend: `FaceRecognizer could not be initialized`
- Confirm `ml/embeddings/embeddings.pkl` exists.
- Run `cd ml && .venv/bin/python src/generate_embeddings.py` to regenerate.

### Backend: `ml_ready: false` in health check
- Confirm both ONNX model files exist in `ml/models/`.
- Check the backend terminal for the specific error.

### Login fails with 401
- Use the correct capitalisation: `Teacher@123`, `Security@123`, `Student@123`, `Admin@123`.

### Excel download produces an error
- Ensure you are logged in as **teacher** or **admin** (not student or security).
- Confirm attendance records exist in the selected date range.

---

## Known Limitations

| Limitation | Notes |
|------------|-------|
| Roll numbers auto-seeded as `emb-Name` | Must be updated to real enrollment IDs by an admin |
| Department and Class default to "Unassigned" | Must be updated via database or an admin API |
| No class schedule / timetable | The system records actuals; absent students only appear when a teacher marks them |
| No admin management UI | Admin must use the REST API or SQLite browser directly |
| No login rate limiting | Add `slowapi` before production deployment |
| SQLite | Not suitable for > 20 concurrent users; migrate to PostgreSQL for production |
| No HTTPS locally | Camera API requires HTTPS on non-localhost origins in production |
| Webcam not tested in CI | Live recognition requires physical interaction |

---

## Future Improvements

- [ ] Admin dashboard for managing students, roll numbers, and departments
- [ ] Timetable/schedule integration for automatic absent marking
- [ ] Multi-angle liveness detection to prevent photo spoofing
- [ ] PostgreSQL support for production deployments
- [ ] Docker / docker-compose for consistent deployment
- [ ] Email/SMS notifications for low-attendance students
- [ ] Exportable PDF reports alongside Excel
- [ ] CI/CD pipeline with automated tests on pull requests

---

## Security & Privacy

- **No raw face images** are stored in the database. Only 128-dimensional embedding vectors are stored in `embeddings.pkl`.
- **Embeddings are biometric data.** The `embeddings.pkl` file and `ml/dataset/registered/` are excluded from Git by `.gitignore`. Keep them private.
- **The SQLite database** contains student names, attendance records, and guest information. It is excluded from Git.
- **Passwords** are hashed with bcrypt (cost factor 12) and never stored in plain text.
- **JWT tokens** are signed with HS256 and expire after 24 hours by default.
- In production, **always** use HTTPS, set a strong `JWT_SECRET_KEY`, and change all development passwords.
- This system is intended for internal institutional use. Comply with applicable data-protection regulations (e.g. DPDPA, GDPR) before deploying with real student data.

---

## License

This project is submitted as a college academic project. A formal open-source license has not yet been selected. Contact the author before reusing or redistributing.
