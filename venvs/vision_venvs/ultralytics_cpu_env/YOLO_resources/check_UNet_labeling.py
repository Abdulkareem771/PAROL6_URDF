import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ----------------------------
# Resolve project paths
# ----------------------------
current_dir = Path(__file__)
project_dir = current_dir.parent.parent

train_dir = project_dir / "data" / "U-Net_data" / "U-Net_model_v2" / "train"

# ----------------------------
# Get all image files
# ----------------------------
image_files = sorted([
    p for p in train_dir.glob("*.jpg")
    if not p.name.endswith("_mask.png")
])

print(f"Found {len(image_files)} images")

# ----------------------------
# Visualization parameters
# ----------------------------
ALPHA = 0.35           # Mask transparency
MASK_COLOR = (0,255,0) # Green overlay (BGR)
CONTOUR_COLOR = (255,0,0)  # Red contour (BGR)

# ----------------------------
# Loop through dataset
# ----------------------------
for img_path in image_files:

    mask_path = img_path.with_name(img_path.stem + "_mask.png")

    if not mask_path.exists():
        print(f"❌ Missing mask for {img_path.name}")
        continue

    # Load image & mask
    img  = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), 0)

    # Ensure binary mask
    mask_bin = (mask > 0).astype(np.uint8) * 255

    # ----------------------------
    # Create overlay
    # ----------------------------
    overlay = img.copy()
    overlay[mask_bin == 255] = MASK_COLOR

    blended = cv2.addWeighted(img, 1 - ALPHA, overlay, ALPHA, 0)

    # ----------------------------
    # Draw contours (optional but recommended)
    # ----------------------------
    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, CONTOUR_COLOR, 2)

    # ----------------------------
    # Display
    # ----------------------------
    plt.figure(figsize=(6,6))
    plt.imshow(blended[:,:,::-1])
    plt.title(img_path.name)
    plt.axis("off")

    plt.show()
