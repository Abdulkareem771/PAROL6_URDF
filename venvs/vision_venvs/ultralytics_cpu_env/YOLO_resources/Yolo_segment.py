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

if len(obj_matrices) >= 2:
    obj_2 = obj_matrices[1]

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

# 6. GUI Display Section
plt.figure(figsize=(20, 5))

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
