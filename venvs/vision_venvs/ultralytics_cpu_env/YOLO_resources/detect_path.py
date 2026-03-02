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
EPSILON_FACTOR = 0.05
EXPAND_PX     = 0   # pixels to expand the polygon outward from each corner
CEXPAND_PX    = 10  # pixels to dilate each contour mask outward


current_dir = Path(__file__)
project_dir = current_dir.parent.parent

SINGLE_IMAGE = project_dir / "data" / "some_images" / "image_a6.png"

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

    
    def find_contours(mask):
        """Return the outermost (external) contour of the largest object in 'mask'.
        Uses CHAIN_APPROX_NONE to keep every boundary pixel.
        Returns the contour as an array of shape (N, 1, 2), or None if not found."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        # Pick the largest contour (the main object)
        return max(contours, key=cv2.contourArea)


    # Find the full external contours from the original masks
    contour_G = find_contours(G)
    contour_R = find_contours(R)

    #if contour_G is not None:
        #cv2.drawContours(img_annotated, [contour_G], -1, (0, 0, 255), 2)   # blue outline (green object)

    #if contour_R is not None:
        #cv2.drawContours(img_annotated, [contour_R], -1, (0, 0, 255), 2)   # blue outline (red object)

    # Expand contours outward by CEXPAND_PX using morphological dilation
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*CEXPAND_PX+1, 2*CEXPAND_PX+1))
    G_exp = cv2.dilate(G, dil_kernel)
    R_exp = cv2.dilate(R, dil_kernel)

    contour_G_exp = find_contours(G_exp)
    contour_R_exp = find_contours(R_exp)

    #if contour_G_exp is not None:
        #cv2.drawContours(img_annotated, [contour_G_exp], -1, (0, 255, 0), 2)  # green = expanded green contour

    #if contour_R_exp is not None:
        #cv2.drawContours(img_annotated, [contour_R_exp], -1, (255, 0, 0), 2)  # red = expanded red contour

    # Intersection of the two expanded contour regions
    intersection_mask = cv2.bitwise_and(G_exp, R_exp)
    contour_I = find_contours(intersection_mask)

    if contour_I is not None:
        cv2.drawContours(img_annotated, [contour_I], -1, (255, 255, 0), 3)    # yellow = intersection region


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

    return G, R, img_annotated

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

g_matrix, r_matrix, img_annotated, bbox_G, bbox_R, bbox_I, corners_G, corners_R = segment_blocks(SINGLE_IMAGE)





