"""
service.py
----------
Face recognition service wrapper connecting the FastAPI backend to the
existing ML module in ml/src/ without modifying or duplicating ML code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.config import (
    COSINE_THRESHOLD,
    ML_EMBEDDINGS_PATH,
    ML_SFACE_MODEL_PATH,
    ML_YUNET_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# Add ml/src to Python path so we can import the original ML modules directly
_ML_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "src"
if str(_ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_SRC_DIR))

try:
    from face_recognizer import FaceRecognizer, FaceResult
except ImportError as err:
    logger.error("Failed to import face_recognizer from ml/src: %s", err)
    FaceRecognizer = None  # type: ignore
    FaceResult = None  # type: ignore


class MLService:
    """Singleton service for Face Detection and Recognition."""

    _instance: Optional[MLService] = None
    _recognizer: Optional[FaceRecognizer] = None

    def __init__(self) -> None:
        if not ML_EMBEDDINGS_PATH.exists():
            logger.warning(
                "Embeddings file does not exist at %s. Face recognition will fail until generated.",
                ML_EMBEDDINGS_PATH,
            )
        self._load_recognizer()

    def _load_recognizer(self) -> None:
        try:
            if FaceRecognizer is not None and ML_EMBEDDINGS_PATH.exists():
                self._recognizer = FaceRecognizer(
                    embeddings_file=str(ML_EMBEDDINGS_PATH),
                    cosine_threshold=COSINE_THRESHOLD,
                )
                logger.info(
                    "ML FaceRecognizer loaded successfully with threshold %.4f",
                    COSINE_THRESHOLD,
                )
            else:
                logger.warning("ML FaceRecognizer could not be initialized.")
        except Exception as exc:
            logger.exception("Error initializing ML FaceRecognizer: %s", exc)
            self._recognizer = None

    @classmethod
    def get_instance(cls) -> MLService:
        if cls._instance is None:
            cls._instance = MLService()
        return cls._instance

    @property
    def is_ready(self) -> bool:
        return self._recognizer is not None

    def recognize_bytes(
        self,
        image_bytes: bytes,
        active_guests: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decode raw image bytes into BGR format and run multi-identity recognition:
        1. Compare against registered students (embeddings.pkl)
        2. Compare against active approved guests (from database)
        3. Authoritatively classify into:
           - REGISTERED_STUDENT (green box)
           - APPROVED_GUEST (orange box)
           - UNKNOWN_PERSON (red box)
           - INCONCLUSIVE (yellow box)
        """
        if not self._recognizer:
            raise RuntimeError("Face recognition model is not loaded or unavailable.")

        # Decode image bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        if np_arr.size == 0:
            raise ValueError("Uploaded image data is empty.")

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Uploaded file could not be decoded as an image.")

        # Run face detector
        faces = self._recognizer._detector.detect(image)
        if faces is None or len(faces) == 0:
            return [
                {
                    "name": "UNKNOWN",
                    "similarity": 0.0,
                    "status": "NO_FACE",
                    "identity_category": "UNKNOWN_PERSON",
                    "authorization_status": "UNAUTHORIZED",
                    "box_color": "red",
                    "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
                    "detection_score": 0.0,
                    "guest_id": None,
                    "valid_until": None,
                    "label": "No Face Detected",
                }
            ]

        output: List[Dict[str, Any]] = []

        for face_row in faces:
            x, y, w, h = face_row[0:4].astype(int)
            det_score = float(face_row[14])

            # Generate 128-d SFace embedding
            aligned = self._recognizer._sface.alignCrop(image, face_row)
            query_emb = self._recognizer._sface.feature(aligned).flatten().astype(np.float32)

            # ── 1. Compare against registered students ────────────────────────
            best_student_name = "UNKNOWN"
            best_student_sim = 0.0

            for person, emb_list in self._recognizer._registered.items():
                for reg_emb in emb_list:
                    sim = self._recognizer._cosine_similarity(query_emb, reg_emb)
                    if sim > best_student_sim:
                        best_student_sim = sim
                        best_student_name = person

            # ── 2. Compare against active approved guests ─────────────────────
            best_guest_id = None
            best_guest_name = None
            best_guest_sim = 0.0
            best_guest_valid_until = None

            if active_guests:
                for guest in active_guests:
                    raw_blob = guest.get("face_embedding")
                    if raw_blob:
                        guest_emb = np.frombuffer(raw_blob, dtype=np.float32)
                        sim = self._recognizer._cosine_similarity(query_emb, guest_emb)
                        if sim > best_guest_sim:
                            best_guest_sim = sim
                            best_guest_name = guest.get("full_name")
                            best_guest_id = guest.get("guest_id")
                            best_guest_valid_until = guest.get("valid_until")

            threshold = self._recognizer.cosine_threshold

            # ── 3. Multi-Identity Classification & Authorization Decision ────
            if best_student_sim >= threshold and best_student_sim >= best_guest_sim:
                # Matched registered student
                identity_category = "REGISTERED_STUDENT"
                auth_status = "AUTHORIZED"
                box_color = "green"
                status_str = "RECOGNISED"
                name = best_student_name
                sim_score = best_student_sim
                label_str = f"Student: {name} ({int(round(sim_score * 100))}%)"
                g_id = None
                valid_until_str = None

            elif best_guest_sim >= threshold and best_guest_sim > best_student_sim:
                # Matched approved guest with active validity
                identity_category = "APPROVED_GUEST"
                auth_status = "AUTHORIZED"
                box_color = "orange"
                status_str = "RECOGNISED"
                name = best_guest_name or "Guest"
                sim_score = best_guest_sim
                label_str = f"Approved Guest: {name} ({int(round(sim_score * 100))}%)"
                g_id = best_guest_id
                valid_until_str = best_guest_valid_until

            else:
                # Inconclusive vs Unknown
                sim_score = max(best_student_sim, best_guest_sim)
                if det_score < 0.70 or w < 45 or h < 45:
                    identity_category = "INCONCLUSIVE"
                    auth_status = "UNVERIFIED"
                    box_color = "yellow"
                    status_str = "INCONCLUSIVE"
                    name = "UNVERIFIED"
                    label_str = "Unverified / Check Required"
                else:
                    identity_category = "UNKNOWN_PERSON"
                    auth_status = "UNAUTHORIZED"
                    box_color = "red"
                    status_str = "UNKNOWN"
                    name = "UNKNOWN PERSON"
                    label_str = f"Unknown Person ({int(round(sim_score * 100))}%)"

                g_id = None
                valid_until_str = None

            output.append(
                {
                    "name": name,
                    "similarity": round(float(sim_score), 4),
                    "status": status_str,
                    "identity_category": identity_category,
                    "authorization_status": auth_status,
                    "box_color": box_color,
                    "bbox": {"x": max(0, int(x)), "y": max(0, int(y)), "w": int(w), "h": int(h)},
                    "detection_score": round(float(det_score), 4),
                    "guest_id": g_id,
                    "valid_until": valid_until_str,
                    "label": label_str,
                }
            )

        return output

    def extract_guest_face_template(
        self, image_bytes: bytes
    ) -> tuple[np.ndarray, str, Dict[str, int]]:
        """
        Extract a verified 128-d SFace embedding and thumbnail for a new guest registration.

        Validation rules:
        - Image must contain exactly 1 face.
        - Detection score must be >= 0.70.
        - Returns (embedding_array, thumbnail_data_url, bbox_dict).
        """
        if not self._recognizer:
            raise RuntimeError("Face recognition model is not loaded or unavailable.")

        np_arr = np.frombuffer(image_bytes, np.uint8)
        if np_arr.size == 0:
            raise ValueError("Uploaded guest face image is empty.")

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Uploaded file could not be decoded as a valid image.")

        faces = self._recognizer._detector.detect(image)
        if faces is None or len(faces) == 0:
            raise ValueError("No face detected in submitted sample. Please provide a clear facial photo.")

        if len(faces) > 1:
            raise ValueError(
                f"Ambiguous sample: {len(faces)} faces detected. Please provide an image with exactly one face."
            )

        face_row = faces[0]
        det_score = float(face_row[14])
        if det_score < 0.70:
            raise ValueError(
                f"Face image quality is too low (confidence: {det_score:.2f}). Please provide a clearer photo."
            )

        # Generate 128-d SFace embedding
        aligned = self._recognizer._sface.alignCrop(image, face_row)
        emb = self._recognizer._sface.feature(aligned).flatten().astype(np.float32)

        # Crop face thumbnail for security audit
        x, y, w, h = face_row[0:4].astype(int)
        ih, iw = image.shape[:2]
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.25)
        crop_x1 = max(0, x - pad_x)
        crop_y1 = max(0, y - pad_y)
        crop_x2 = min(iw, x + w + pad_x)
        crop_y2 = min(ih, y + h + pad_y)
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size > 0:
            crop_resized = cv2.resize(crop, (120, 120))
            _, buf = cv2.imencode(".jpg", crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            import base64
            thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
        else:
            thumb_b64 = ""

        bbox_dict = {"x": max(0, int(x)), "y": max(0, int(y)), "w": int(w), "h": int(h)}
        return emb, thumb_b64, bbox_dict
