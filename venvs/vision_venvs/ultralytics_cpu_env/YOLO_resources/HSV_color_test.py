import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

y_min_G, y_max_G = 0, 0
x_min_G, x_max_G = 0, 0
y_min_R, y_max_R = 0, 0
x_min_R, x_max_R = 0, 0
EPSILON_FACTOR = 0.05
EXPAND_PX     = 0   # pixels to expand the polygon outward from each corner
CEXPAND_PX    = 12  # pixels to dilate each contour mask outward


current_dir = Path(__file__)                # YOLO_resources/detect_path.py
project_dir = current_dir.parent.parent     # ultralytics_cpu_env

SINGLE_IMAGE = project_dir / "data" / "some_images" / "annotated_image.png"

IMAGE_FOLDER = project_dir / "data" / "Segmentation_images"

def segment_blocks(image_path):
    # 1. Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return None, None

    # Convert BGR to RGB for correct display in Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Convert BGR (OpenCV default) to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. Define color ranges in HSV
    # Note: HSV ranges in OpenCV are H: 0-180, S: 0-255, V: 0-255
    
    # Red range (Red wraps around 0 and 180, so we combine two ranges)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    

    # 4. Create Color Mask (as a matrix)
    
    # R matrix: combine both ends of the red spectrum
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    R = cv2.bitwise_or(mask_red1, mask_red2)
    px_num = np.where((R < 255) & (R > 0))
    px_num_y = px_num[0]
    print(f"px_num: {px_num}")
    #px_num_255 = np.where(R == 255)
    #print(f"px_num_255: {px_num_255}")
    

    # Optional: Clean up noise with morphological operations
    kernel = np.ones((3, 3), np.uint8)

    #R_2 = cv2.morphologyEx(R, cv2.MORPH_OPEN, kernel)
    R_eroded = cv2.erode(R,  kernel, iterations=1)
    R_dilated = cv2.dilate(R, kernel, iterations=2)
    
    
    img_annotated = img_rgb.copy()

    return R, R_eroded, R_dilated, img_annotated, img_rgb


# --- Execution ---
r_mask, r_eroded, r_dilated, img_annotated, img_rgb = segment_blocks(SINGLE_IMAGE)



#plt.figure(figsize=(20, 20))

# Subplot 1: Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')


"""
# Subplot 2: G Matrix (Green Mask)
plt.subplot(1, 2, 1)
plt.title("Green Object")
plt.imshow(g_matrix, cmap='gray')
plt.axis('off')

# Subplot 3: B Matrix (Blue Object)
plt.subplot(1, 2, 2)
plt.title("Blue Object")
plt.imshow(b_matrix, cmap='gray')
plt.axis('off')
"""

# Subplot 1: R Matrix (Red Object)
plt.subplot(1, 2, 2)
plt.title("Red Object")
plt.imshow(r_mask, cmap='gray')
plt.axis('off')

"""
# Subplot 2: R eroded
plt.subplot(1, 3, 2)
plt.title("Red Object Eroded")
plt.imshow(r_eroded, cmap='gray')
plt.axis('off')


# Subplot 3: R dilated
plt.subplot(1, 3, 3)
plt.title("Red Object Dilated")
plt.imshow(r_dilated)
plt.axis('off')
"""
plt.tight_layout()
plt.show()



