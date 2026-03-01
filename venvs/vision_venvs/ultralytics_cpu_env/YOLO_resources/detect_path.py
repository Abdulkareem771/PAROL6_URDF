import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

y_min_G, y_max_G = 0, 0
x_min_G, x_max_G = 0, 0
y_min_R, y_max_R = 0, 0
x_min_R, x_max_R = 0, 0




current_dir = Path(__file__)
project_dir = current_dir.parent.parent

SINGLE_IMAGE = project_dir / "data" / "some_images" / "image_2.jpg"

IMAGE_FOLDER = project_dir / "data" / "Segmentation_images"

def segment_blocks(image_path):
    # 1. Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None, None

    # Convert BGR to RGB for correct display in Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Convert BGR (OpenCV default) to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Define color ranges in HSV
    # Note: HSV ranges in OpenCV are H: 0-180, S: 0-255, V: 0-255
    
    # Green range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Red range (Red wraps around 0 and 180, so we combine two ranges)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 4. Create Masks (G and R matrices)
    # G matrix: 255 for green pixels, 0 otherwise
    G = cv2.inRange(hsv, lower_green, upper_green)

    # R matrix: combine both ends of the red spectrum
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    R = cv2.bitwise_or(mask_red1, mask_red2)

    # Optional: Clean up noise with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    G = cv2.morphologyEx(G, cv2.MORPH_OPEN, kernel)
    R = cv2.morphologyEx(R, cv2.MORPH_OPEN, kernel)

    # 5. Compute Bounding Boxes and draw on a copy of img_rgb
    img_annotated = img_rgb.copy()

    # Green bounding box
    r_G, c_G = np.where(G == 255)
    if len(r_G) > 0:
        y_min_G, y_max_G = int(r_G.min()), int(r_G.max())
        x_min_G, x_max_G = int(c_G.min()), int(c_G.max())
        y_min_G = y_min_G - 2
        y_max_G = y_max_G + 2
        x_min_G = x_min_G - 2
        x_max_G = x_max_G + 2
        cv2.rectangle(img_annotated, (x_min_G, y_min_G), (x_max_G, y_max_G), (0, 0, 255), 2)
    else:
        x_min_G = x_max_G = y_min_G = y_max_G = 0

    # Red bounding box
    r_R, c_R = np.where(R == 255)
    if len(r_R) > 0:
        y_min_R, y_max_R = int(r_R.min()), int(r_R.max())
        x_min_R, x_max_R = int(c_R.min()), int(c_R.max())
        y_min_R = y_min_R - 2
        y_max_R = y_max_R + 2
        x_min_R = x_min_R - 2
        x_max_R = x_max_R + 2
        cv2.rectangle(img_annotated, (x_min_R, y_min_R), (x_max_R, y_max_R), (0, 0, 255), 2)
    else:
        x_min_R = x_max_R = y_min_R = y_max_R = 0

    # 6. Compute Intersection of the two bounding boxes
    inter_x_min = max(x_min_G, x_min_R)
    inter_y_min = max(y_min_G, y_min_R)
    inter_x_max = min(x_max_G, x_max_R)
    inter_y_max = min(y_max_G, y_max_R)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        # Boxes overlap — draw intersection region in yellow
        cv2.rectangle(img_annotated, (inter_x_min, inter_y_min), (inter_x_max, inter_y_max), (255, 255, 0), 2)
        bbox_I = (inter_x_min, inter_y_min, inter_x_max, inter_y_max)
    else:
        bbox_I = None
        print("No intersection between the two bounding boxes.")

    # 7. GUI Display Section
    plt.figure(figsize=(20, 5))

    # Subplot 1: Original Image
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    # Subplot 2: G Matrix (Green Mask)
    plt.subplot(1, 4, 2)
    plt.title("G Matrix (Green Object)")
    plt.imshow(G, cmap='gray')
    plt.axis('off')

    # Subplot 3: R Matrix (Red Object)
    plt.subplot(1, 4, 3)
    plt.title("R Matrix (Red Object)")
    plt.imshow(R, cmap='gray')
    plt.axis('off')

    # Subplot 4: Annotated Image with Bounding Boxes
    plt.subplot(1, 4, 4)
    plt.title("Seam Path")
    plt.imshow(img_annotated)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    bbox_G = (x_min_G, y_min_G, x_max_G, y_max_G)
    bbox_R = (x_min_R, y_min_R, x_max_R, y_max_R)
    return G, R, img_annotated, bbox_G, bbox_R, bbox_I

def process_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supported extensions
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    for img_path in image_files:
        filename = os.path.basename(img_path).split('.')[0]
        G, R = segment_blocks(img_path)

        if G is not None:
            # Save the masks as images (or keep as matrices for further ops)
            cv2.imwrite(os.path.join(output_folder, f"{filename}_mask_G.png"), G)
            cv2.imwrite(os.path.join(output_folder, f"{filename}_mask_R.png"), R)
            print(f"Processed: {filename}")


# --- Execution ---
#process_folder('input_folder_path', 'output_folder_path')
# Replace 'image.jpg' with your file or use the folder function

g_matrix, r_matrix, img_annotated, bbox_G, bbox_R, bbox_I = segment_blocks(SINGLE_IMAGE)

x_min_G, y_min_G, x_max_G, y_max_G = bbox_G
x_min_R, y_min_R, x_max_R, y_max_R = bbox_R

# Width and height of Green object:
w_G = x_max_G - x_min_G
h_G = y_max_G - y_min_G

# Width and height of Red object:
w_R = x_max_R - x_min_R
h_R = y_max_R - y_min_R

print(f"Green Object Bounding Box: ({x_min_G}, {y_min_G}) to ({x_max_G}, {y_max_G})")
print(f"Green Object width: {w_G}, height: {h_G}")

print(f"Red Object Bounding Box:   ({x_min_R}, {y_min_R}) to ({x_max_R}, {y_max_R})")
print(f"Red Object width: {w_R}, height: {h_R}")

# Intersection region
if bbox_I is not None:
    x_min_I, y_min_I, x_max_I, y_max_I = bbox_I
    w_I = x_max_I - x_min_I
    h_I = y_max_I - y_min_I
    print(f"\nIntersection Region:       ({x_min_I}, {y_min_I}) to ({x_max_I}, {y_max_I})")
    print(f"Intersection width: {w_I}, height: {h_I}")
else:
    print("\nNo intersection between the two bounding boxes.")


