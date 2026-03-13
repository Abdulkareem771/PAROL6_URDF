"""
Design an interactive GUI with Controls as following:
→ / D  : Next image
← / A  : Previous image
S      : Save current annotated frame
T      : Auto-save annotated frames for ALL images
Q / ESC: Quit
"""


from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

CEXPAND_PX = 8

# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "Presentation_Samples"
OUTPUT_DIR = PROJECT_DIR / "data" / "Presentation_Samples" / "model_v2"

MODEL_PATH_v1 = PROJECT_DIR / "yolo_training" / "experiment_12_YOLO_Segmentation" / "weights" / "best.pt"
MODEL_PATH_v2 = PROJECT_DIR / "yolo_segmentation_models_results" / "experiment_2" / "weights" / "best.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH_v2)

# -----------------------------
# Collect Images
# -----------------------------
image_paths = sorted(list(DATA_DIR.rglob("*.jpg")) +
                     list(DATA_DIR.rglob("*.png")) +
                     list(DATA_DIR.rglob("*.jpeg")))

print("Images found:", len(image_paths))

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
    
    
    if contour_obj1 is not None:
        cv2.drawContours(annotated, [contour_obj1], -1, (255,0,0), 4)

    if contour_obj2 is not None:
        cv2.drawContours(annotated, [contour_obj2], -1, (255,0,0), 4)
    
    
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
    """
    if contour_I is not None:
        cv2.drawContours(annotated, [contour_I], -1, (0,0,255), -1)
        #cv2.drawContours(annotated, [contour_I], -1, (255,255,0), 3)
    """
    return annotated



def save_all_images(image_paths):

    print("\nStarting automatic saving of all annotated images...\n")

    for i, img_path in enumerate(image_paths):

        annotated = process_image(img_path)

        if annotated is None:
            continue

        save_path = OUTPUT_DIR / f"{img_path.stem}_annotated.png"

        cv2.imwrite(str(save_path), annotated)

        print(f"[{i+1}/{len(image_paths)}] Saved:", save_path)

    print("\nFinished saving all images.\n")


def resize_to_screen(img, max_w=1400, max_h=900):
    """
    Resize image to fit inside the screen while keeping aspect ratio.
    """
    h, w = img.shape[:2]

    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h, 1.0)   # never upscale

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h))


cv2.namedWindow("YOLO Segmentation Viewer", cv2.WINDOW_NORMAL)
# -----------------------------
# GUI LOOP
# -----------------------------
index = 0
total = len(image_paths)

while True:

    image_path = image_paths[index]

    annotated = process_image(image_path)

    display = annotated.copy()

    text = f"{index+1}/{total} : {image_path.name}"
    cv2.putText(display, text, (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,0), 2)

    display_resized = resize_to_screen(display)
    cv2.imshow("YOLO Segmentation Viewer", display_resized)

    key = cv2.waitKey(0) & 0xFF

    
    # Next image
    if key in [ord('d'), 83]:
        index = min(index + 1, total-1)

    # Previous image
    elif key in [ord('a'), 81]:
        index = max(index - 1, 0)

    # Save image
    elif key == ord('s'):

        save_path = OUTPUT_DIR / f"{image_path.stem}_annotated.png"
        cv2.imwrite(str(save_path), annotated)

        print("Saved:", save_path)

    # 🔵 Auto-save ALL images
    elif key == ord('t'):

        confirm = input("Process ALL images? (y/n): ")

        if confirm.lower() == "y":
            save_all_images(image_paths)
    
    # Quit
    elif key in [ord('q'), 27]:
        break


cv2.destroyAllWindows()