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

PROCESS_MODE = "folder"     # "single" or "folder"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
CEXPAND_PX    = 10  # pixels to dilate each contour mask outward


# --- Configuration ---
# Adjust these paths to match your project structure

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "YOLO_Segmentation_data" / "test" 
OUTPUT_DIR = PROJECT_DIR / "data" / "YOLO_Segmentation_results"
SINGLE_IMAGE = PROJECT_DIR / "data" / "YOLO_Segmentation_data" / "test" / "6.jpg"

# Model paths
MODEL_PATH = PROJECT_DIR / "yolo_segmentation_models_results" / "experiment_2" / "weights" / "best.pt"

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH)


def process_image(image_path):

    print(f"\nProcessing: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    h, w = img.shape[:2]

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
    masks = result.masks.data if result.masks is not None else None

    obj_matrices = []

    if masks is not None:
        for mask in masks:

            mask_np = mask.cpu().numpy()

            mask_resized = cv2.resize(mask_np, (w, h))

            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            obj_matrices.append(mask_binary)

    if len(obj_matrices) < 2:
        print("Less than two objects detected")
        return

    obj_1 = obj_matrices[0]
    obj_2 = obj_matrices[1]

    # -----------------------------
    # Morphological cleanup
    # -----------------------------
    kernel = np.ones((5, 5), np.uint8)
    obj_1 = cv2.morphologyEx(obj_1, cv2.MORPH_OPEN, kernel)
    obj_2 = cv2.morphologyEx(obj_2, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # Contours
    # -----------------------------
    def find_contours(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    contour_obj1 = find_contours(obj_1)
    contour_obj2 = find_contours(obj_2)

    if contour_obj1 is not None:
        cv2.drawContours(img_annotated, [contour_obj1], -1, (0, 0, 255), 2)

    if contour_obj2 is not None:
        cv2.drawContours(img_annotated, [contour_obj2], -1, (0, 0, 255), 2)

    # -----------------------------
    # Contour Expansion
    # -----------------------------
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2*CEXPAND_PX+1, 2*CEXPAND_PX+1)
    )

    obj_1_exp = cv2.dilate(obj_1, dil_kernel)
    obj_2_exp = cv2.dilate(obj_2, dil_kernel)

    contour_obj1_exp = find_contours(obj_1_exp)
    contour_obj2_exp = find_contours(obj_2_exp)

    if contour_obj1_exp is not None:
        cv2.drawContours(img_annotated, [contour_obj1_exp], -1, (0, 255, 0), 2)

    if contour_obj2_exp is not None:
        cv2.drawContours(img_annotated, [contour_obj2_exp], -1, (0, 255, 0), 2)

    # -----------------------------
    # Intersection
    # -----------------------------
    intersection_mask = cv2.bitwise_and(obj_1_exp, obj_2_exp)
    contour_I = find_contours(intersection_mask)

    if contour_I is not None:
        cv2.drawContours(img_annotated, [contour_I], -1, (255, 0, 0), -1)
        cv2.drawContours(img_annotated, [contour_I], -1, (255, 255, 0), 3)

    # -----------------------------
    # Display
    # -----------------------------
    plt.figure(figsize=(20, 20))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Object 1 Mask")
    plt.imshow(obj_1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Object 2 Mask")
    plt.imshow(obj_2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Seam Path")
    plt.imshow(img_annotated)
    plt.axis('off')

    plt.tight_layout()
    plt.show()



# -----------------------------
# Main Execution
# -----------------------------

if PROCESS_MODE == "single":

    process_image(SINGLE_IMAGE)

elif PROCESS_MODE == "folder":

    image_files = []

    for ext in IMAGE_EXTENSIONS:
        image_files.extend(DATA_DIR.rglob(f"*{ext}"))

    print(f"Found {len(image_files)} images")

    for img_path in image_files:
        process_image(img_path)


