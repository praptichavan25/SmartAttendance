#!/usr/bin/env python3
"""
run.py
------
Convenience runner to launch the SmartAttendance FastAPI backend.
Usage:
    python run.py
"""

import sys
from pathlib import Path

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import uvicorn
from app.config import BACKEND_HOST, BACKEND_PORT

if __name__ == "__main__":
    print(f"Starting SmartAttendance Backend on http://{BACKEND_HOST}:{BACKEND_PORT}...")
    uvicorn.run("app.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
