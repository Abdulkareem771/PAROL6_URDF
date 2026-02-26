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

image_files = sorted([
    p for p in train_dir.glob("*.jpg")
    if not p.name.endswith("_mask.png")
])

num_images = len(image_files)
print(f"Found {num_images} images")
print("Controls: [d]=Next | [a]=Prev | [q]=Quit")

# ----------------------------
# Visualization parameters
# ----------------------------
ALPHA = 0.35
MASK_COLOR = (0, 255, 0)
CONTOUR_COLOR = (255, 0, 0)

idx = 0
quit_flag = False

# ----------------------------
# Key handler
# ----------------------------
def on_key(event):
    global idx, quit_flag

    if event.key == 'q':
        quit_flag = True
        plt.close()
    elif event.key == 'd':
        idx = min(idx + 1, num_images - 1)
        plt.close()
    elif event.key == 'a':
        idx = max(idx - 1, 0)
        plt.close()
    else:
        # Any other key → forward
        idx = min(idx + 1, num_images - 1)
        plt.close()

# ----------------------------
# Main loop
# ----------------------------
while not quit_flag:

    img_path = image_files[idx]
    mask_path = img_path.with_name(img_path.stem + "_mask.png")

    if not mask_path.exists():
        print(f"❌ Missing mask for {img_path.name}")
        idx = min(idx + 1, num_images - 1)
        continue

    # Load image & mask
    img  = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), 0)

    mask_bin = (mask > 0).astype(np.uint8) * 255

    # Create overlay
    overlay = img.copy()
    overlay[mask_bin == 255] = MASK_COLOR
    blended = cv2.addWeighted(img, 1 - ALPHA, overlay, ALPHA, 0)

    # Draw contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, CONTOUR_COLOR, 2)

    # Display
    fig = plt.figure(figsize=(6, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.imshow(blended[:, :, ::-1])
    plt.title(f"{idx+1}/{num_images}  —  {img_path.name}")
    plt.axis("off")

    plt.show()

print("🛑 Viewer closed")