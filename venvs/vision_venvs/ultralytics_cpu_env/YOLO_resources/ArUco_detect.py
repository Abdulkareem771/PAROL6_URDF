import cv2
import cv2.aruco as aruco
from pathlib import Path

current_dir = Path(__file__)                # YOLO_resources/detect_path.py
project_dir = current_dir.parent.parent     # ultralytics_cpu_env

SINGLE_IMAGE = project_dir / "data" / "some_images" / "aruco-marker-ID=6.png"

# -----------------------------
# Select ArUco dictionary
# -----------------------------
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Detector parameters
parameters = aruco.DetectorParameters()

# Create detector
detector = aruco.ArucoDetector(ARUCO_DICT, parameters)

# -----------------------------
# Start video capture
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit")

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (recommended)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # Draw detected markers
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Print IDs
        for i, marker_id in enumerate(ids):
            print(f"Detected ID: {marker_id[0]}")

    # Show frame
    cv2.imshow("ArUco Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()