#!/usr/bin/env python3
"""
Phase 2 - Mode 1: YOLO Segmentation Prediction GUI Viewer
==========================================================
Loads the trained YOLO segmentation model and runs predictions on a folder
of images. Displays each image with its predicted segmentation masks in an
interactive OpenCV GUI.

Controls:
  →  / D       : Next image
  ←  / A       : Previous image
  +  / =       : Increase mask opacity
  -            : Decrease mask opacity
  M            : Toggle mask overlay on / off
  B            : Toggle bounding boxes on / off
  L            : Toggle class labels on / off
  C            : Toggle confidence scores on / off
  I            : Toggle seam-intersection overlay on / off
  S            : Save current annotated frame to disk
  R            : Re-run prediction on current image (useful after conf change)
  Q / ESC      : Quit

Configuration variables are at the top of the CONFIGURATION section below.
"""

import cv2
import numpy as np
import os
import random
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# ---- Paths ------------------------------------------------------------------
current_dir   = Path(__file__).resolve().parent
project_dir   = current_dir.parent                        # ultralytics_cpu_env

# Trained YOLO segmentation model
MODEL_PATH = project_dir / "yolo_segmentation_models_results" / "experiment_2" / "weights" / "best.pt"

# Folder that contains the images to predict on.
# Change this to any folder with images you want to test.
#IMAGE_FOLDER = project_dir / "data" / "raw_images_for_models"
IMAGE_FOLDER = project_dir / "data" / "YOLO_Segmentation_data" / "test"
# If IMAGE_FOLDER does not exist or is empty the script falls back to the
# YOLO Segmentation test split (used during training validation).
FALLBACK_IMAGE_FOLDER = project_dir / "data" / "YOLO_Segmentation_data" / "test"

# Where saved annotated frames are written
OUTPUT_FOLDER = project_dir / "data" / "phase2_predictions"

# ---- Prediction settings ----------------------------------------------------
CONFIDENCE_THRESHOLD = 0.45   # minimum confidence to show a detection
IOU_THRESHOLD        = 0.45   # NMS IoU threshold
DEVICE               = "cpu"  # "cpu" or "cuda" (or e.g. "cuda:0")

# ---- Display settings -------------------------------------------------------
MASK_ALPHA_INIT  = 0.45       # initial mask overlay transparency (0.0 – 1.0)
ALPHA_STEP       = 0.05       # how much + / - changes the alpha
WINDOW_NAME      = "YOLO Segmentation Viewer  |  ← →:navigate  M:mask  I:seam  T:seam-mode  S:save  Q:quit"
WINDOW_WIDTH     = 1280       # initial window width  (resizable)
WINDOW_HEIGHT    = 720        # initial window height (resizable)

# ---- Seam-intersection (Image_processing_first_mode port) -------------------
# Class IDs that your YOLO model uses for the two objects whose boundary
# intersection defines the seam.  Class 0 is typically the first class listed
# in your dataset.yaml, class 1 the second.  Adjust as needed.
CLASS_ID_A  = 0    # e.g. "green block" / first  object class
CLASS_ID_B  = 1    # e.g. "blue block"  / second object class

# Pixels to dilate each object mask/bbox outward before computing intersection.
# Larger values → wider overlap zone captured.
CEXPAND_PX  = 50

# Seam-intersection mode: how the per-class regions are built.
#   "mask"  – use the YOLO segmentation polygon masks  (Mode 1 – masking)
#   "bbox"  – use the YOLO bounding-box rectangles      (Mode 2 – bboxing)
# Can be toggled at runtime with the T key.
SEAM_MODE   = "mask"   # default startup mode

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# =============================================================================
# HELPERS
# =============================================================================

def load_images(folder: Path) -> list[Path]:
    """Return a sorted list of image paths found in *folder*."""
    if not folder.exists():
        return []
    files = [p for p in sorted(folder.iterdir())
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return files


def random_color(seed: int) -> tuple[int, int, int]:
    """Return a deterministic, vivid BGR colour keyed by *seed* (class id)."""
    rng = random.Random(seed * 7919 + 13)
    h   = rng.randint(0, 179)
    hsv = np.uint8([[[h, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# =============================================================================
# SEAM-INTERSECTION HELPERS  (ported from Image_processing_first_mode.py)
# =============================================================================

def find_largest_contour(binary_mask: np.ndarray):
    """Return the largest external contour found in *binary_mask* (uint8, 0/255).

    Uses CHAIN_APPROX_NONE to keep every boundary pixel, identical to the
    approach in Image_processing_first_mode.py.  Returns the contour array
    (shape N×1×2) or None if no contour is found.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def build_class_mask(result, class_id: int, img_h: int, img_w: int) -> np.ndarray:
    """Build a binary uint8 mask (0 / 255) that is the union of all YOLO
    segmentation masks belonging to *class_id* in *result*.

    The YOLO masks are stored as polygon coordinates (result.masks.xy).
    We fill them onto a blank canvas the same size as the original image.
    """
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    if result.masks is None:
        return canvas

    class_ids  = result.boxes.cls.cpu().numpy().astype(int)
    masks_poly  = result.masks.xy  # list of (N,2) float arrays (pixel coords)

    for poly, cid in zip(masks_poly, class_ids):
        if cid != class_id or len(poly) == 0:
            continue
        pts = poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], 255)

    return canvas


def compute_seam_intersection(
    result,
    img_h: int,
    img_w: int,
    class_id_a: int  = CLASS_ID_A,
    class_id_b: int  = CLASS_ID_B,
    expand_px:  int  = CEXPAND_PX,
):
    """MODE 1 – MASKING: intersection of the dilated segmentation-polygon masks.

    Returns
    -------
    contour_I  : largest contour of the intersection region, or None.
    mask_A     : raw binary mask for class A (before dilation)
    mask_B     : raw binary mask for class B (before dilation)
    mask_A_exp : dilated mask for class A
    mask_B_exp : dilated mask for class B
    intersection : raw binary intersection mask (0/255)
    """
    # 1. Build per-class binary masks from the YOLO polygon predictions
    mask_A = build_class_mask(result, class_id_a, img_h, img_w)
    mask_B = build_class_mask(result, class_id_b, img_h, img_w)

    # 2. Dilate each mask outward by expand_px
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1)
    )
    mask_A_exp = cv2.dilate(mask_A, dil_kernel)
    mask_B_exp = cv2.dilate(mask_B, dil_kernel)

    # 3. Compute the intersection of the two expanded masks
    intersection = cv2.bitwise_and(mask_A_exp, mask_B_exp)

    # 4. Find the largest contour in the intersection region
    contour_I = find_largest_contour(intersection)

    return contour_I, mask_A, mask_B, mask_A_exp, mask_B_exp, intersection


# =============================================================================
# SEAM-INTERSECTION MODE 2 – BBOXING
# =============================================================================

def build_class_mask_from_bbox(
    result,
    class_id: int,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Build a binary uint8 mask (0 / 255) by filling the bounding-box rectangle
    of every detection that belongs to *class_id*.

    This is the bbox equivalent of build_class_mask(); instead of the precise
    segmentation polygon, it uses the axis-aligned bounding box.
    """
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    if result.boxes is None:
        return canvas

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # shape (N, 4)

    for box, cid in zip(boxes_xyxy, class_ids):
        if cid != class_id:
            continue
        x1, y1, x2, y2 = map(int, box)
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, cv2.FILLED)

    return canvas


def compute_seam_intersection_bbox(
    result,
    img_h: int,
    img_w: int,
    class_id_a: int  = CLASS_ID_A,
    class_id_b: int  = CLASS_ID_B,
    expand_px:  int  = CEXPAND_PX,
):
    """MODE 2 – BBOXING: intersection of the dilated bounding-box rectangles.

    Identical pipeline to compute_seam_intersection(), but uses filled
    bounding-box rectangles instead of segmentation polygons as the source
    masks.  Useful when mask quality is low.

    Returns the same six values as compute_seam_intersection().
    """
    # 1. Build per-class rectangle masks from the YOLO bounding boxes
    mask_A = build_class_mask_from_bbox(result, class_id_a, img_h, img_w)
    mask_B = build_class_mask_from_bbox(result, class_id_b, img_h, img_w)

    # 2. Dilate each rectangle mask outward by expand_px
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1)
    )
    mask_A_exp = cv2.dilate(mask_A, dil_kernel)
    mask_B_exp = cv2.dilate(mask_B, dil_kernel)

    # 3. Intersection of the two expanded bbox masks
    intersection = cv2.bitwise_and(mask_A_exp, mask_B_exp)

    # 4. Find the largest contour of the intersection region
    contour_I = find_largest_contour(intersection)

    return contour_I, mask_A, mask_B, mask_A_exp, mask_B_exp, intersection


def draw_predictions(
    image: np.ndarray,
    result,
    mask_alpha:    float = 0.45,
    show_masks:    bool  = True,
    show_boxes:    bool  = True,
    show_labels:   bool  = True,
    show_conf:     bool  = True,
    show_seam:     bool  = True,
    seam_mode:     str   = "mask",   # "mask" (Mode 1) or "bbox" (Mode 2)
) -> tuple[np.ndarray, np.ndarray]:
    """Render YOLO segmentation result onto *image*.

    Returns
    -------
    annotated : BGR image with masks, boxes, labels, and seam overlay drawn on it.
    seam_path : Nx2 int32 array of [x, y] pixel coordinates that belong to the
                intersection region (empty array when show_seam is False or no
                intersection exists).
    Parameters
    ----------
    image      : BGR image (H × W × 3)
    result     : ultralytics Results object (single image)
    mask_alpha : transparency of filled polygon masks
    show_*     : toggle individual visual elements
    """
    annotated = image.copy()
    h, w      = annotated.shape[:2]

    if result.masks is None:
        # No detections → show original with a "No detections" banner
        cv2.putText(annotated, "No detections", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2, cv2.LINE_AA)
        return annotated, np.empty((0, 2), dtype=np.int32)

    # Pre-compute seam intersection (needs full image size; done once per frame)
    seam_contour  = None
    seam_path     = np.empty((0, 2), dtype=np.int32)   # Nx2 array of [x, y] coords
    if show_seam:
        # Dispatch to the correct mode
        if seam_mode == "bbox":
            seam_contour, _, _, _, _, seam_mask = compute_seam_intersection_bbox(result, h, w)
        else:   # default: "mask"
            seam_contour, _, _, _, _, seam_mask = compute_seam_intersection(result, h, w)
        # seam_path: every pixel inside the intersection region as [x, y] pairs
        ys, xs = np.where(seam_mask > 0)
        seam_path = np.column_stack((xs, ys)).astype(np.int32) if len(xs) > 0 else np.empty((0, 2), dtype=np.int32)

    boxes       = result.boxes          # Boxes object
    masks_data  = result.masks.xy       # list of (N,2) polygon arrays in pixel coords
    class_ids   = boxes.cls.cpu().numpy().astype(int)
    confs       = boxes.conf.cpu().numpy()
    names       = result.names          # dict {id: name}

    # ---- 1. Mask overlay (filled polygons) ----------------------------------
    if show_masks:
        overlay = annotated.copy()
        for poly, cls_id in zip(masks_data, class_ids):
            if len(poly) == 0:
                continue
            color = random_color(cls_id)
            pts   = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, mask_alpha, annotated, 1 - mask_alpha, 0, annotated)

    # ---- 2. Mask contours ---------------------------------------------------
    if show_masks:
        for poly, cls_id in zip(masks_data, class_ids):
            if len(poly) == 0:
                continue
            color = random_color(cls_id)
            pts   = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

    # ---- 3. Bounding boxes --------------------------------------------------
    if show_boxes:
        for box, cls_id, conf in zip(boxes.xyxy.cpu().numpy(), class_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            color = random_color(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label above the box
            if show_labels or show_conf:
                label_parts = []
                if show_labels:
                    label_parts.append(names.get(cls_id, str(cls_id)))
                if show_conf:
                    label_parts.append(f"{conf:.2f}")
                label = " ".join(label_parts)

                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                lx, ly = x1, max(y1 - 6, th + baseline)
                cv2.rectangle(annotated, (lx, ly - th - baseline),
                              (lx + tw + 4, ly + baseline), color, cv2.FILLED)
                cv2.putText(annotated, label, (lx + 2, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                            1, cv2.LINE_AA)

    # ---- 4. Seam intersection: filled red region + solid red outline ---------
    # Drawn last so it is always visible on top of all other annotations.
    if show_seam and seam_contour is not None:
        # 4a. Semi-transparent red fill over the intersection area
        seam_overlay = annotated.copy()
        cv2.drawContours(seam_overlay, [seam_contour], -1, (0, 0, 255), cv2.FILLED)
        cv2.addWeighted(seam_overlay, 0.45, annotated, 0.55, 0, annotated)
        # 4b. Solid red outline on top (3 px, same as Image_processing script)
        cv2.drawContours(annotated, [seam_contour], -1, (0, 0, 255), 3)

    return annotated, seam_path


def draw_hud(
    annotated:   np.ndarray,
    idx:         int,
    total:       int,
    image_name:  str,
    num_det:     int,
    mask_alpha:  float,
    show_masks:  bool,
    show_boxes:  bool,
    show_labels: bool,
    show_conf:   bool,
    show_seam:   bool  = True,
    seam_mode:   str   = "mask",
) -> np.ndarray:
    """Overlay a status bar at the bottom of the frame."""
    frame  = annotated.copy()
    h, w   = frame.shape[:2]
    bar_h  = 36
    bar_y  = h - bar_h

    # Dark translucent strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status text
    flags = []
    flags.append(f"Mask:{'ON' if show_masks else 'OFF'}")
    flags.append(f"Box:{'ON' if show_boxes else 'OFF'}")
    flags.append(f"Label:{'ON' if show_labels else 'OFF'}")
    flags.append(f"Conf:{'ON' if show_conf else 'OFF'}")
    flags.append(f"Seam:{'ON' if show_seam else 'OFF'}")
    flags.append(f"SeamMode:{'BBOX' if seam_mode == 'bbox' else 'MASK'}")
    flags.append(f"α={mask_alpha:.2f}")

    left_text  = f"[{idx+1}/{total}]  {image_name}   Detections: {num_det}"
    right_text = "  |  ".join(flags) + "   S=save  Q=quit"

    cv2.putText(frame, left_text, (10, bar_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(frame, right_text, (w - rw - 10, bar_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1, cv2.LINE_AA)

    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    from ultralytics import YOLO

    # ---- Validate model path ------------------------------------------------
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found:\n  {MODEL_PATH}")
        sys.exit(1)

    print(f"[INFO] Loading model from:\n  {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print("[INFO] Model loaded successfully.")

    # ---- Collect images -----------------------------------------------------
    image_files = load_images(IMAGE_FOLDER)
    if not image_files:
        print(f"[WARN] No images found in: {IMAGE_FOLDER}")
        print(f"[INFO] Falling back to: {FALLBACK_IMAGE_FOLDER}")
        image_files = load_images(FALLBACK_IMAGE_FOLDER)

    if not image_files:
        print("[ERROR] No images found. Please update IMAGE_FOLDER in the script.")
        sys.exit(1)

    print(f"[INFO] Found {len(image_files)} image(s) to process.")

    # ---- Prepare output folder ----------------------------------------------
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # ---- State --------------------------------------------------------------
    idx         = 0
    mask_alpha  = MASK_ALPHA_INIT
    show_masks  = True
    show_boxes  = True
    show_labels = True
    show_conf   = True
    show_seam   = True
    seam_mode   = SEAM_MODE          # "mask" or "bbox" — toggle with T

    # Cache: dict[int -> (original_bgr, result)]
    cache: dict[int, tuple[np.ndarray, object]] = {}

    def get_prediction(i: int):
        """Return (original_image, result) for index *i*, using cache."""
        if i not in cache:
            img_path = image_files[i]
            img_bgr  = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[WARN] Could not load image: {img_path}")
                img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            results = model.predict(
                source=img_bgr,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )
            cache[i] = (img_bgr, results[0])
        return cache[i]

    # ---- OpenCV window ------------------------------------------------------
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    print("\n[INFO] GUI launched. Controls:")
    print("  ← / → (or A / D) : navigate images")
    print("  M                : toggle mask overlay")
    print("  B                : toggle bounding boxes")
    print("  L                : toggle class labels")
    print("  C                : toggle confidence scores")
    print("  I                : toggle seam-intersection overlay")
    print("  T                : toggle seam mode  [MASK = use seg. polygons | BBOX = use bounding boxes]")
    print(f"                     (CLASS_ID_A={CLASS_ID_A}, CLASS_ID_B={CLASS_ID_B}, CEXPAND_PX={CEXPAND_PX})")
    print("  + / -            : adjust mask opacity")
    print("  S                : save current annotated image")
    print("  R                : rerun prediction (clears cache for this image)")
    print("  Q / ESC          : quit\n")

    while True:
        # Clamp index
        idx = max(0, min(idx, len(image_files) - 1))

        # Run / retrieve prediction
        orig_bgr, result = get_prediction(idx)
        num_det = len(result.boxes) if result.boxes is not None else 0

        # Render
        annotated, seam_path = draw_predictions(
            orig_bgr, result,
            mask_alpha=mask_alpha,
            show_masks=show_masks,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_conf=show_conf,
            show_seam=show_seam,
            seam_mode=seam_mode,
        )
        if seam_path.size > 0:
            print(f"[SEAM/{seam_mode.upper()}] seam_path: {len(seam_path)} pixels  "
                  f"| x∈[{seam_path[:,0].min()}, {seam_path[:,0].max()}]  "
                  f"y∈[{seam_path[:,1].min()}, {seam_path[:,1].max()}]")
        frame = draw_hud(
            annotated,
            idx=idx,
            total=len(image_files),
            image_name=image_files[idx].name,
            num_det=num_det,
            mask_alpha=mask_alpha,
            show_masks=show_masks,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_conf=show_conf,
            show_seam=show_seam,
            seam_mode=seam_mode,
        )

        cv2.imshow(WINDOW_NAME, frame)

        # ---- Key handling ---------------------------------------------------
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:          # Q or ESC → quit
            break

        elif key == ord('d') or key == 83:        # D or →  → next
            idx += 1

        elif key == ord('a') or key == 81:        # A or ←  → previous
            idx -= 1

        elif key == ord('m'):                     # M → toggle masks
            show_masks = not show_masks

        elif key == ord('b'):                     # B → toggle boxes
            show_boxes = not show_boxes

        elif key == ord('l'):                     # L → toggle labels
            show_labels = not show_labels

        elif key == ord('c'):                     # C → toggle confidence
            show_conf = not show_conf

        elif key == ord('i'):                     # I → toggle seam intersection
            show_seam = not show_seam

        elif key == ord('t'):                     # T → toggle seam mode mask ↔ bbox
            seam_mode = "bbox" if seam_mode == "mask" else "mask"
            print(f"[INFO] Seam mode switched to: {seam_mode.upper()}")

        elif key in (ord('+'), ord('=')):         # + → more opaque
            mask_alpha = min(1.0, mask_alpha + ALPHA_STEP)

        elif key == ord('-'):                     # - → more transparent
            mask_alpha = max(0.0, mask_alpha - ALPHA_STEP)

        elif key == ord('s'):                     # S → save frame
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem      = image_files[idx].stem
            out_path  = OUTPUT_FOLDER / f"{stem}_pred_{ts}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"[SAVE] Annotated image saved → {out_path}")

        elif key == ord('r'):                     # R → re-run prediction
            if idx in cache:
                del cache[idx]
            print(f"[INFO] Re-running prediction on: {image_files[idx].name}")

        # Check if window was closed by the user
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("[INFO] Viewer closed. Goodbye!")


if __name__ == "__main__":
    main()
