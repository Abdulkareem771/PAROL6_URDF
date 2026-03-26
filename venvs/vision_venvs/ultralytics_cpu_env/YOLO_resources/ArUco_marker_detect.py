import cv2
import numpy as np
from pathlib import Path

current_dir = Path(__file__)                # YOLO_resources/detect_path.py
project_dir = current_dir.parent.parent     # ultralytics_cpu_env

SINGLE_IMAGE = project_dir / "data" / "some_images" / "aruco-marker-ID=6.png"



# Load your image
img = cv2.imread(str(SINGLE_IMAGE))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# List of dictionaries to test
aruco_dicts = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_4X4_1000,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_5X5_100,
    cv2.aruco.DICT_5X5_250,
    cv2.aruco.DICT_5X5_1000,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_6X6_100,
    cv2.aruco.DICT_6X6_250,
    cv2.aruco.DICT_6X6_1000,
    cv2.aruco.DICT_7X7_50,
    cv2.aruco.DICT_7X7_100,
    cv2.aruco.DICT_7X7_250,
    cv2.aruco.DICT_7X7_1000,
    cv2.aruco.DICT_ARUCO_ORIGINAL
]

for dict_id in aruco_dicts:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        print(f"Detected with dictionary: {dict_id}")
        print(f"Marker IDs: {ids.flatten()}")