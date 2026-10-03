import cv2


def main():
    # Open MacBook's default built-in webcam (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        print("Please ensure camera permissions are granted in macOS System Settings:")
        print("System Settings -> Privacy & Security -> Camera")
        return

    print("Webcam successfully opened!")
    print("Press 'q' in the camera window to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Display the live camera stream
        cv2.imshow("MacBook Webcam Test - Press 'q' to Quit", frame)

        # Check for 'q' key press to close
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break

    # Clean up and release camera resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and window closed.")


if __name__ == "__main__":
    main()
