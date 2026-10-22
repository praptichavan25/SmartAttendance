"""
main.py
-------
FastAPI entrypoint for the SmartAttendance application backend.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.config import BACKEND_HOST, BACKEND_PORT, CORS_ORIGIN
from app.db.database import init_db
from app.ml.service import MLService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smartattendance")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI: initializes DB and loads ML models on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Pre-warming ML models...")
    _ = MLService.get_instance()
    logger.info("Backend application startup complete.")
    yield
    logger.info("Backend application shutting down.")


app = FastAPI(
    title="SmartAttendance Backend API",
    description="Backend API for SmartAttendance facial recognition and attendance tracking.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS setup — strict origin whitelist for local development
allowed_origins = [
    CORS_ORIGIN,
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
# Remove duplicates and empty strings
allowed_origins = list({orig.strip() for orig in allowed_origins if orig.strip()})

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root():
    return {
        "service": "SmartAttendance Backend API",
        "status": "online",
        "documentation": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
