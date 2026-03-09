"""
YOLO Instance Segmentation → Extract Pixel Matrices

This script:
1. Loads a trained YOLO segmentation model (best.pt)
2. Runs inference on an image
3. Extracts the segmentation mask of each detected object
4. Converts each mask into a numpy matrix with the same shape as the original image
5. Stores them as obj_1, obj_2, ...

Author: Auto-generated
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


CEXPAND_PX    = 10  # pixels to dilate each contour mask outward


# --- Configuration ---
# Adjust these paths to match your project structure

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "YOLO_Segmentation_data"
OUTPUT_DIR = PROJECT_DIR / "data" / "YOLO_Segmentation_results"
SINGLE_IMAGE = PROJECT_DIR / "data" / "YOLO_Segmentation_data" / "test" / "26.jpg"

# Model paths
MODEL_PATH = PROJECT_DIR / "yolo_segmentation_models_results" / "experiment_2" / "weights" / "best.pt"

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Load Image
# -----------------------------
img = cv2.imread(str(SINGLE_IMAGE))
h, w = img.shape[:2]
if img is None:
    print(f"Could not read image: {SINGLE_IMAGE}")
    exit()

# Convert BGR to RGB for correct display in Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img_annotated = img_rgb.copy()

# -----------------------------
# Run Inference
# -----------------------------
results = model(img)

result = results[0]
#print(f"Number of objects detected: {len(result)}")

# -----------------------------
# Extract Masks
# -----------------------------
masks = result.masks.data  # tensor: (num_objects, mask_h, mask_w)

obj_matrices = []

if masks is not None:

    for i, mask in enumerate(masks):

        # Convert to numpy
        mask_np = mask.cpu().numpy()

        # Resize mask to original image size
        mask_resized = cv2.resize(mask_np, (w, h))

        # Convert to binary (0 or 255)
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        obj_matrices.append(mask_binary)

# -----------------------------
# Assign objects
# -----------------------------
if len(obj_matrices) >= 1:
    obj_1 = obj_matrices[0]
    coords_obj1 = np.column_stack(np.where(obj_1 == 255))

if len(obj_matrices) >= 2:
    obj_2 = obj_matrices[1]
    coords_obj2 = np.column_stack(np.where(obj_2 == 255))

#print(f"Shape of coords_obj1: {coords_obj1.shape}")
#print(f"Shape of coords_obj2: {coords_obj2.shape}")

# Optional: Clean up noise with morphological operations
kernel = np.ones((5, 5), np.uint8)
obj_1 = cv2.morphologyEx(obj_1, cv2.MORPH_OPEN, kernel)
obj_2 = cv2.morphologyEx(obj_2, cv2.MORPH_OPEN, kernel)


def find_contours(mask):
        """Return the outermost (external) contour of the largest object in 'mask'.
        Uses CHAIN_APPROX_NONE to keep every boundary pixel.
        Returns the contour as an array of shape (N, 1, 2), or None if not found."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        # Pick the largest contour (the main object)
        return max(contours, key=cv2.contourArea)

# Find the full external contours from the original masks
contour_obj1 = find_contours(obj_1)
contour_obj2 = find_contours(obj_2)
"""
if contour_obj1 is not None:
    cv2.drawContours(img_annotated, [contour_obj1], -1, (0, 0, 255), 2)   # blue outline (First object)

if contour_obj2 is not None:
    cv2.drawContours(img_annotated, [contour_obj2], -1, (0, 0, 255), 2)   # blue outline (Second object)
"""
# Expand contours outward by CEXPAND_PX using morphological dilation
dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*CEXPAND_PX+1, 2*CEXPAND_PX+1))
obj_1_exp = cv2.dilate(obj_1, dil_kernel)
obj_2_exp = cv2.dilate(obj_2, dil_kernel)

contour_obj1_exp = find_contours(obj_1_exp)
contour_obj2_exp = find_contours(obj_2_exp)

"""
if contour_obj1_exp is not None:
    cv2.drawContours(img_annotated, [contour_obj1_exp], -1, (0, 255, 0), 2)  # green = expanded object 1 contour

if contour_obj2_exp is not None:
    cv2.drawContours(img_annotated, [contour_obj2_exp], -1, (255, 0, 0), 2)  # red = expanded object 2 contour
"""
# Intersection of the two expanded contour regions
intersection_mask = cv2.bitwise_and(obj_1_exp, obj_2_exp)
contour_I = find_contours(intersection_mask)

if contour_I is not None:
    cv2.drawContours(img_annotated, [contour_I], -1, (255, 0, 0), -1)    # red = intersection region (filled)
    cv2.drawContours(img_annotated, [contour_I], -1, (255, 0, 0), 3)    # red = intersection region (outline)





# -----------------------------
# Example Outputs
# -----------------------------
print("Number of objects detected:", len(obj_matrices))

if len(obj_matrices) >= 1:
    print("obj_1 shape:", obj_1.shape)

if len(obj_matrices) >= 2:
    print("obj_2 shape:", obj_2.shape)

"""
# Optional: visualize
if len(obj_matrices) >= 1:
    cv2.imshow("Object 1 Mask", obj_1)

if len(obj_matrices) >= 2:
    cv2.imshow("Object 2 Mask", obj_2)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""



plt.figure(figsize=(20, 20))

plt.title("Seam Path")
plt.imshow(img_annotated)
plt.axis('off')
plt.tight_layout()
plt.show()


"""
# 6. GUI Display Section
plt.figure(figsize=(20, 20))

# Subplot 1: Original Image
plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')

# Subplot 2: G Matrix (Green Mask)
plt.subplot(1, 4, 2)
plt.title("Object 1 Mask")
plt.imshow(obj_1, cmap='gray')
plt.axis('off')

# Subplot 3: R Matrix (Red Object)
plt.subplot(1, 4, 3)
plt.title("Object 2 Mask")
plt.imshow(obj_2, cmap='gray')
plt.axis('off')

# Subplot 4: Annotated Image with Bounding Boxes
plt.subplot(1, 4, 4)
plt.title("Seam Path")
plt.imshow(img_annotated)
plt.axis('off')

plt.tight_layout()
plt.show()
"""