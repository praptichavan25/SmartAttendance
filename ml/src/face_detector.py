"""
face_detector.py
----------------
Reusable wrapper around OpenCV's YuNet face detector.

Reuses the existing YuNet ONNX model already present at ml/models/.
Compatible with cv2.FaceRecognizerSF (SFace) which expects the same
bounding-box format that YuNet outputs.
"""

import os
import urllib.request
import cv2

# --------------------------------------------------------------------------- #
# Model location & download
# --------------------------------------------------------------------------- #

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
YUNET_MODEL_PATH = os.path.join(_ML_ROOT, "models", "face_detection_yunet_2023mar.onnx")


def ensure_yunet_model() -> str:
    """Return the path to the YuNet model, downloading it if necessary."""
    if not os.path.exists(YUNET_MODEL_PATH):
        os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
        print(f"Downloading YuNet model → {YUNET_MODEL_PATH}")
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
    return YUNET_MODEL_PATH


# --------------------------------------------------------------------------- #
# Detector class
# --------------------------------------------------------------------------- #

class YuNetDetector:
    """
    Thin wrapper around cv2.FaceDetectorYN (YuNet).

    Parameters
    ----------
    score_threshold : float
        Minimum confidence to keep a detection (default 0.8).
    nms_threshold : float
        Non-maximum suppression threshold (default 0.3).
    top_k : int
        Maximum candidate detections before NMS (default 5000).
    """

    def __init__(
        self,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        model_path = ensure_yunet_model()
        self._detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),  # updated dynamically per image
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, image: "cv2.Mat"):
        """
        Detect faces in a BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image loaded with cv2.imread.

        Returns
        -------
        faces : np.ndarray | None
            Array of shape (N, 15) where each row is
            [x, y, w, h, lm0_x, lm0_y, …, lm4_x, lm4_y, score],
            or None when no faces are detected.
        """
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        return faces
