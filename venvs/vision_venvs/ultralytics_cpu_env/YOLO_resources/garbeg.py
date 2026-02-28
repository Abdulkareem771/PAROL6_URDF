import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


current_dir = Path(__file__)
project_dir = current_dir.parent.parent

SINGLE_IMAGE = project_dir / "data" / "some_images" / "image.jpg"

IMAGE_FOLDER = project_dir / "data" / "Segmentation_images"


def segment_and_display(image_path):
    # 1. Load the image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert BGR to RGB for correct display in Matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Convert to HSV for robust segmentation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 3. Define Color Thresholds
    # Green
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Red (Red wraps around 0 and 180 in HSV)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])

    # 4. Generate G and R Matrices
    G = cv2.inRange(hsv, lower_green, upper_green)
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    R = cv2.bitwise_or(mask_red1, mask_red2)

    # 5. GUI Display Section
    plt.figure(figsize=(15, 5))

    # Subplot 1: Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    # Subplot 2: G Matrix (Green Mask)
    plt.subplot(1, 3, 2)
    plt.title("G Matrix (Green Object)")
    plt.imshow(G, cmap='gray')
    plt.axis('off')

    # Subplot 3: R Matrix (Red Object)
    plt.subplot(1, 3, 3)
    plt.title("R Matrix (Red Object)")
    plt.imshow(R, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return G, R

# Run the function
# Replace 'image.jpg' with your actual file path
g_matrix, r_matrix = segment_and_display(SINGLE_IMAGE)