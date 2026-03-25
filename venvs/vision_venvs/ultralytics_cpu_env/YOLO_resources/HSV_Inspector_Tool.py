import cv2
import numpy as np
import os
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
current_dir = Path(__file__)                # YOLO_resources/detect_path.py
project_dir = current_dir.parent.parent     # ultralytics_cpu_env

SINGLE_IMAGE = project_dir / "data" / "some_images" / "annotated_image.png"


IMAGE_PATH = SINGLE_IMAGE
WINDOW_NAME = "HSV Inspector"

# Global
img_bgr = None
img_hsv = None
current_display = None

# -----------------------------
# Mouse callback
# -----------------------------
def mouse_callback(event, x, y, flags, param):
    global current_display

    if event == cv2.EVENT_MOUSEMOVE:
        if x >= img_bgr.shape[1] or y >= img_bgr.shape[0]:
            return

        b, g, r = img_bgr[y, x]
        h, s, v = img_hsv[y, x]

        display = img_bgr.copy()

        # -----------------------------
        # Draw vertical reference line (centered example)
        # -----------------------------
        cx = img_bgr.shape[1] // 2
        cv2.line(display, (cx, 0), (cx, img_bgr.shape[0]), (0, 0, 255), 2)

        # -----------------------------
        # Draw small yellow circle at cursor
        # -----------------------------
        cv2.circle(display, (x, y), 5, (0, 255, 255), -1)

        # -----------------------------
        # Overlay text (top-left like your screenshot)
        # -----------------------------
        cv2.putText(display, f"Pixel: ({x}, {y})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display, f"HSV: ({h}, {s}, {v})", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(display, f"RGB: ({r}, {g}, {b})", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        current_display = display

# -----------------------------
# Main
# -----------------------------
def main():
    global img_bgr, img_hsv, current_display

    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("Error: Could not load image")
        return

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    current_display = img_bgr.copy()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        cv2.imshow(WINDOW_NAME, current_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
