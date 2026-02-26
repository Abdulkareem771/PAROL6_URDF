import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Resolve project paths
# ----------------------------
current_dir = Path(__file__)
project_dir = current_dir.parent.parent

train_dir = project_dir / "data" / "U-Net_data" / "U-Net_model.v1i.png-mask-semantic" / "train"

# ----------------------------
# Get all image files (exclude masks)
# ----------------------------
image_files = sorted([
    p for p in train_dir.glob("*.jpg")
    if not p.name.endswith("_mask.png")
])

print(f"Found {len(image_files)} images")

# ----------------------------
# Loop through images
# ----------------------------
for img_path in image_files:

    mask_path = img_path.with_name(
        img_path.stem + "_mask.png"
    )

    if not mask_path.exists():
        print(f"❌ Missing mask for {img_path.name}")
        continue

    # Load image & mask
    img  = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), 0)

    # Plot side-by-side
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img[:,:,::-1])
    plt.title(f"Image: {img_path.name}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()