import cv2
import numpy as np

# ---------- CONFIG ----------
IMAGE_PATH = "image.jpg"   # <-- Change this to your image path
# ----------------------------

# Load image (OpenCV loads in BGR)
image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    print("Error: Could not load image.")
    exit()

# Convert BGR -> RGB -> HSV
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Copy for drawing text
display = image_bgr.copy()

def mouse_callback(event, x, y, flags, param):
    global display

    if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
        # Get pixel values
        b, g, r = image_bgr[y, x]
        h, s, v = image_hsv[y, x]

        # Reset image
        display = image_bgr.copy()

        # Text to show
        text1 = f"Pixel: ({x}, {y})"
        text2 = f"RGB: ({r}, {g}, {b})"
        text3 = f"HSV: ({h}, {s}, {v})"

        # Draw text on image
        cv2.putText(display, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(display, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(display, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Draw small circle on cursor
        cv2.circle(display, (x, y), 5, (0, 255, 0), 2)

# Create window and attach callback
cv2.namedWindow("HSV Inspector")
cv2.setMouseCallback("HSV Inspector", mouse_callback)

# Main loop
while True:
    cv2.imshow("HSV Inspector", display)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
