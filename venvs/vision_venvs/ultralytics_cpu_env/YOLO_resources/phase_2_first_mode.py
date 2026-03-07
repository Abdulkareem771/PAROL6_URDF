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
MODEL_PATH = project_dir / "yolo_training" / "experiment_12_YOLO_Segmentation" / "weights" / "best.pt"

# Folder that contains the images to predict on.
# Change this to any folder with images you want to test.
IMAGE_FOLDER = project_dir / "data" / "YOLO_Segmentation_data" / "test"

# If IMAGE_FOLDER does not exist or is empty the script falls back to the
# YOLO Segmentation test split (used during training validation).
FALLBACK_IMAGE_FOLDER = project_dir / "data" / "YOLO_Segmentation_data" / "test"

# Where saved annotated frames are written
OUTPUT_FOLDER = project_dir / "data" / "phase2_predictions"

# ---- Prediction settings ----------------------------------------------------
CONFIDENCE_THRESHOLD = 0.25   # minimum confidence to show a detection
IOU_THRESHOLD        = 0.45   # NMS IoU threshold
DEVICE               = "cpu"  # "cpu" or "cuda" (or e.g. "cuda:0")

# ---- Display settings -------------------------------------------------------
MASK_ALPHA_INIT  = 0.45       # initial mask overlay transparency (0.0 – 1.0)
ALPHA_STEP       = 0.05       # how much + / - changes the alpha
WINDOW_NAME      = "YOLO Segmentation Viewer  |  ← →:navigate  M:mask  B:bbox  S:save  Q:quit"
WINDOW_WIDTH     = 1280       # initial window width  (resizable)
WINDOW_HEIGHT    = 720        # initial window height (resizable)

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


def draw_predictions(
    image: np.ndarray,
    result,
    mask_alpha:    float = 0.45,
    show_masks:    bool  = True,
    show_boxes:    bool  = True,
    show_labels:   bool  = True,
    show_conf:     bool  = True,
) -> np.ndarray:
    """
    Render YOLO segmentation result onto *image* and return the annotated copy.

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
        return annotated

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

    return annotated


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
        annotated = draw_predictions(
            orig_bgr, result,
            mask_alpha=mask_alpha,
            show_masks=show_masks,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_conf=show_conf,
        )
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
