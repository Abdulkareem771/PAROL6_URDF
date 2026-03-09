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
image = cv2.imread(IMAGE_PATH)
h, w = image.shape[:2]

# -----------------------------
# Run Inference
# -----------------------------
results = model(image)

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

# Optional: visualize
if len(obj_matrices) >= 1:
    cv2.imshow("Object 1 Mask", obj_1)

if len(obj_matrices) >= 2:
    cv2.imshow("Object 2 Mask", obj_2)

cv2.waitKey(0)
cv2.destroyAllWindows()