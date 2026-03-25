import cv2
import numpy as np

# Load image
image = cv2.imread("input.jpg")
if image is None:
    raise FileNotFoundError("Image not found!")

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Red color ranges (HSV)
lower_red_1 = np.array([0, 120, 70])
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([170, 120, 70])
upper_red_2 = np.array([180, 255, 255])

# Create masks
mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

red_mask = mask1 | mask2

# Extract red pixels
red_pixels = cv2.bitwise_and(image, image, mask=red_mask)

# Save outputs
cv2.imwrite("red_mask.png", red_mask)
cv2.imwrite("red_pixels.png", red_pixels)

print("✅ Red pixel detection completed")
print("📁 Saved: red_mask.png, red_pixels.png")
