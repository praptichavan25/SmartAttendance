import os
import sys
import urllib.request
import cv2

# URL for OpenCV's official YuNet face detection model
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def get_model_path():
    """Locate or download the YuNet model in the ml/models directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "models", "face_detection_yunet_2023mar.onnx"),
        os.path.join(current_dir, "..", "models", "face_detection_yunet_2023mar.onnx"),
        os.path.join(current_dir, "face_detection_yunet_2023mar.onnx"),
    ]

    for path in candidates:
        norm_path = os.path.normpath(path)
        if os.path.exists(norm_path):
            return norm_path

    # Fallback to ml/models/face_detection_yunet_2023mar.onnx
    target_path = os.path.normpath(
        os.path.join(current_dir, "models", "face_detection_yunet_2023mar.onnx")
    )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"Downloading YuNet face detection model to {target_path}...")
    urllib.request.urlretrieve(YUNET_MODEL_URL, target_path)
    return target_path


def main():
    model_path = get_model_path()
    if not os.path.exists(model_path):
        print(f"Error: Face detection model not found at {model_path}", file=sys.stderr)
        return

    # Initialize OpenCV's dedicated YuNet deep-learning face detector
    # score_threshold=0.8 filters out weak detections and avoids false positives
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),  # Will be dynamically updated per frame
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000,
    )

    # Open MacBook's default built-in webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.", file=sys.stderr)
        print("Please ensure camera permissions are granted in macOS System Settings:")
        print("System Settings -> Privacy & Security -> Camera")
        return

    print("Camera and YuNet Face Detector initialized successfully!")
    print("Press 'q' in the camera window to exit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Error: Failed to read frame from webcam.", file=sys.stderr)
                break

            height, width = frame.shape[:2]

            # Update detector input size to match frame dimensions
            detector.setInputSize((width, height))

            # Detect faces using the dedicated neural network
            _, faces = detector.detect(frame)

            face_count = 0
            if faces is not None:
                face_count = len(faces)
                for face in faces:
                    # Bounding box coordinates: [x, y, w, h]
                    x, y, w, h = face[0:4].astype(int)
                    x, y = max(0, x), max(0, y)

                    # Draw bounding box around detected human face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display face count on video
            text = f"Faces detected: {face_count}"
            text_color = (0, 255, 0) if face_count > 0 else (0, 0, 255)
            cv2.putText(
                frame,
                text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                text_color,
                2,
                cv2.LINE_AA,
            )

            # Display live camera stream
            cv2.imshow("Face Detection (YuNet) - Press 'q' to Quit", frame)

            # Exit cleanly when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting face detection...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and window closed.")


if __name__ == "__main__":
    main()
