from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

CEXPAND_PX = 8

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw_images_for_models"
SINGLE_IMAGE_PATH = PROJECT_DIR / "data" / "raw_images_for_models" / "13.jpg"
OUTPUT_DIR = PROJECT_DIR / "data" / "Phase_2_first_mode" / "model_v2"


MODEL_PATH_v1 = PROJECT_DIR / "yolo_training" / "experiment_12_YOLO_Segmentation" / "weights" / "best.pt"
MODEL_PATH_v2 = PROJECT_DIR / "yolo_segmentation_models_results" / "experiment_2" / "weights" / "best.pt"

#OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH_v2)


# -----------------------------
# Contour Function
# -----------------------------
def find_contours(mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


# -----------------------------
# Image Processing Function
# -----------------------------
def process_image(image_path):

    img = cv2.imread(str(image_path))

    if img is None:
        return None

    h, w = img.shape[:2]
    annotated = img.copy()

    results = model(img)
    result = results[0]

    if result.masks is None:
        return img

    masks = result.masks.data

    obj_matrices = []

    for mask in masks:

        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        obj_matrices.append(mask_binary)

    if len(obj_matrices) < 2:
        return img

    obj_1 = obj_matrices[0]
    obj_2 = obj_matrices[1]

    kernel = np.ones((5,5), np.uint8)
    obj_1 = cv2.morphologyEx(obj_1, cv2.MORPH_OPEN, kernel)
    obj_2 = cv2.morphologyEx(obj_2, cv2.MORPH_OPEN, kernel)

    contour_obj1 = find_contours(obj_1)
    contour_obj2 = find_contours(obj_2)
    
    """
    if contour_obj1 is not None:
        cv2.drawContours(annotated, [contour_obj1], -1, (255,0,0), 2)

    if contour_obj2 is not None:
        cv2.drawContours(annotated, [contour_obj2], -1, (255,0,0), 2)
    """
    
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2*CEXPAND_PX+1, 2*CEXPAND_PX+1)
    )

    obj_1_exp = cv2.dilate(obj_1, dil_kernel)
    obj_2_exp = cv2.dilate(obj_2, dil_kernel)

    contour_obj1_exp = find_contours(obj_1_exp)
    contour_obj2_exp = find_contours(obj_2_exp)
    
    """
    if contour_obj1_exp is not None:
        cv2.drawContours(annotated, [contour_obj1_exp], -1, (0,255,0), 2)

    if contour_obj2_exp is not None:
        cv2.drawContours(annotated, [contour_obj2_exp], -1, (0,255,0), 2)
    """

    #intersection_mask = cv2.bitwise_and(obj_1, obj_2)   # Intersection mask without expand
    intersection_mask = cv2.bitwise_and(obj_1_exp, obj_2_exp)  # Intersection mask with expand
    contour_I = find_contours(intersection_mask)

    if contour_I is not None:
        cv2.drawContours(annotated, [contour_I], -1, (0,0,255), -1)
        #cv2.drawContours(annotated, [contour_I], -1, (255,255,0), 3)

    return annotated


annotated = process_image(SINGLE_IMAGE_PATH)

if annotated is not None:
    cv2.imshow("YOLO Segmentation Viewer", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


