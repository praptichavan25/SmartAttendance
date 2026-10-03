"""
generate_embeddings.py
----------------------
Generates face embeddings for every registered person and saves them to
ml/embeddings/embeddings.pkl.

Usage
-----
    # From the ml/ directory (recommended):
    python src/generate_embeddings.py

    # Or with explicit paths:
    python src/generate_embeddings.py \
        --dataset  ml/dataset/registered \
        --out      ml/embeddings/embeddings.pkl

Re-run this script any time new photos are added to ml/dataset/registered/.

Pipeline
--------
1. YuNet (cv2.FaceDetectorYN)     – detect face bounding box in each image
2. SFace (cv2.FaceRecognizerSF)   – align face crop & produce 128-d embedding
3. Embeddings saved as a dict     – {person_name: [embedding, ...]}

Model choices
-------------
- YuNet  : already used in the project; lightweight ONNX detector, no pip deps.
- SFace  : OpenCV's official companion recogniser for YuNet; accepts the exact
           bounding-box format YuNet outputs; no additional pip packages needed;
           pretrained on a large face dataset, produces robust 128-d embeddings.
"""

import argparse
import os
import pickle
import sys
import urllib.request

import cv2
import numpy as np

# Import the reusable YuNet wrapper from the same src/ package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_detector import YuNetDetector

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

DEFAULT_DATASET_DIR = os.path.join(_ML_ROOT, "dataset", "registered")
DEFAULT_OUT_FILE = os.path.join(_ML_ROOT, "embeddings", "embeddings.pkl")

SFACE_MODEL_PATH = os.path.join(_ML_ROOT, "models", "face_recognition_sface_2021dec.onnx")
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

def ensure_sface_model() -> str:
    """Download the SFace ONNX model if not already present."""
    if not os.path.exists(SFACE_MODEL_PATH):
        os.makedirs(os.path.dirname(SFACE_MODEL_PATH), exist_ok=True)
        print(f"Downloading SFace model → {SFACE_MODEL_PATH} …")
        urllib.request.urlretrieve(SFACE_MODEL_URL, SFACE_MODEL_PATH)
        print("  Download complete.")
    return SFACE_MODEL_PATH


# --------------------------------------------------------------------------- #
# Embedding generation
# --------------------------------------------------------------------------- #

def generate_embeddings(dataset_dir: str, out_file: str) -> None:
    """
    Walk dataset_dir/<person_name>/<image> and produce one embedding per face.

    Parameters
    ----------
    dataset_dir : str
        Root of the registered dataset, e.g. ml/dataset/registered/
    out_file : str
        Destination pickle file, e.g. ml/embeddings/embeddings.pkl
    """
    # ── Load models ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("SmartAttendance — Face Embedding Generator")
    print("=" * 60)

    detector = YuNetDetector(score_threshold=0.8)
    print("✓ YuNet face detector loaded (ml/models/face_detection_yunet_2023mar.onnx)")

    sface_path = ensure_sface_model()
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    print(f"✓ SFace recognizer loaded ({os.path.basename(sface_path)})")
    print()

    # ── Scan dataset ─────────────────────────────────────────────────────────
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    person_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    )

    if not person_dirs:
        print("ERROR: No person sub-folders found in dataset directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(person_dirs)} registered person(s): {', '.join(person_dirs)}")
    print()

    embeddings: dict[str, list[np.ndarray]] = {}
    failed_images: list[str] = []

    for person in person_dirs:
        person_dir = os.path.join(dataset_dir, person)
        image_files = sorted(
            f for f in os.listdir(person_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        )

        if not image_files:
            print(f"  [{person}] WARNING: No image files found — skipping.")
            failed_images.append(f"{person}/ (no images)")
            continue

        person_embeddings: list[np.ndarray] = []

        for img_name in image_files:
            img_path = os.path.join(person_dir, img_name)
            rel_path = os.path.join(person, img_name)

            # ── Load image ───────────────────────────────────────────────────
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [{person}] FAILED to read: {img_name}")
                failed_images.append(rel_path + " (unreadable)")
                continue

            # ── Detect face ──────────────────────────────────────────────────
            faces = detector.detect(image)

            if faces is None or len(faces) == 0:
                # Retry with a lower threshold for photos that are slightly far
                # from the camera or have unusual lighting
                detector_relaxed = YuNetDetector(score_threshold=0.6)
                faces = detector_relaxed.detect(image)

            if faces is None or len(faces) == 0:
                print(f"  [{person}] FAILED — no face detected in: {img_name}")
                failed_images.append(rel_path + " (no face detected)")
                continue

            # Use the highest-confidence face (first row after NMS)
            face_box = faces[0]  # shape (15,): [x, y, w, h, lm*5, score]

            # ── Align & embed ────────────────────────────────────────────────
            # alignCrop expects the full face row (including landmarks & score)
            aligned_face = recognizer.alignCrop(image, face_box)
            embedding = recognizer.feature(aligned_face)       # shape (1, 128)
            embedding = embedding.flatten().astype(np.float32) # shape (128,)

            person_embeddings.append(embedding)
            score = float(face_box[14])
            print(f"  [{person}] OK — {img_name}  (detection score: {score:.3f})")

        if person_embeddings:
            embeddings[person] = person_embeddings

    # ── Save ─────────────────────────────────────────────────────────────────
    print()
    print("-" * 60)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"✓ Embeddings saved → {out_file}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("Summary")
    print("─" * 40)
    total = 0
    for person in person_dirs:
        count = len(embeddings.get(person, []))
        total += count
        status = f"{count} embedding(s)" if count > 0 else "FAILED (0 embeddings)"
        print(f"  {person:<15} {status}")

    print()
    print(f"Total embeddings generated : {total}")
    print(f"Total failed images        : {len(failed_images)}")

    if failed_images:
        print()
        print("Failed images:")
        for f in failed_images:
            print(f"  ✗ {f}")
    else:
        print("No failed images. All faces detected successfully.")

    print("=" * 60)


# --------------------------------------------------------------------------- #
# CLI entry-point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate face embeddings from the registered dataset."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_DIR,
        help=f"Path to registered dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_FILE,
        help=f"Output pickle file path (default: {DEFAULT_OUT_FILE})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_embeddings(args.dataset, args.out)
