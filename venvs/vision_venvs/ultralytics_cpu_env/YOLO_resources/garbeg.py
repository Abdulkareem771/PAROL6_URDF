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

    # Green bounding box
    r_G, c_G = np.where(G == 255)
    if len(r_G) > 0:
        y_min_G, y_max_G = int(r_G.min()), int(r_G.max())
        x_min_G, x_max_G = int(c_G.min()), int(c_G.max())
        print(f"Green Object Bounding Box: ({x_min_G}, {y_min_G}) to ({x_max_G}, {y_max_G})")
        y_min_G = y_min_G - EXPAND_PX
        y_max_G = y_max_G + EXPAND_PX
        x_min_G = x_min_G - EXPAND_PX
        x_max_G = x_max_G + EXPAND_PX
        #cv2.rectangle(img_annotated, (x_min_G, y_min_G), (x_max_G, y_max_G), (0, 255, 0), 2)
        # cv2.rectangle replaced below with cv2.polylines after corners are detected
    else:
        x_min_G = x_max_G = y_min_G = y_max_G = 0
    
    # Red bounding box
    r_R, c_R = np.where(R == 255)
    if len(r_R) > 0:
        y_min_R, y_max_R = int(r_R.min()), int(r_R.max())
        x_min_R, x_max_R = int(c_R.min()), int(c_R.max())
        print(f"Red Object Bounding Box: ({x_min_R}, {y_min_R}) to ({x_max_R}, {y_max_R})")
        y_min_R = y_min_R - EXPAND_PX
        y_max_R = y_max_R + EXPAND_PX
        x_min_R = x_min_R - EXPAND_PX
        x_max_R = x_max_R + EXPAND_PX
        #cv2.rectangle(img_annotated, (x_min_R, y_min_R), (x_max_R, y_max_R), (255, 0, 0), 2)
        # cv2.rectangle replaced below with cv2.polylines after corners are detected
    else:
        x_min_R = x_max_R = y_min_R = y_max_R = 0

    
    # 6. Find exact corner coordinates using contour approximation
    def find_corners(mask, epsilon_factor = EPSILON_FACTOR):
        """Return corner points of the largest contour in 'mask' as an (N,2) array of (x,y)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Use the largest contour (main object)
        largest = max(contours, key=cv2.contourArea)
        epsilon = epsilon_factor * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        # approx shape: (N, 1, 2) → reshape to (N, 2)
        return approx.reshape(-1, 2)
    
    
    def find_contours(mask):
        """Return the outermost (external) contour of the largest object in 'mask'.
        Uses CHAIN_APPROX_NONE to keep every boundary pixel.
        Returns the contour as an array of shape (N, 1, 2), or None if not found."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        # Pick the largest contour (the main object)
        return max(contours, key=cv2.contourArea)

    corners_G = find_corners(G)
    corners_R = find_corners(R)

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

    def expand_corners(corners, px):
        """Push each corner outward from the polygon centroid by 'px' pixels."""
        centroid = corners.mean(axis=0)
        direction = corners.astype(np.float32) - centroid
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        expanded = corners.astype(np.float32) + (direction / norms) * px
        return np.round(expanded).astype(np.int32)

    # Draw expanded polygon outline and corner dots on img_annotated
    if corners_G is not None:
        exp_G = expand_corners(corners_G, EXPAND_PX)
        pts_G = exp_G.reshape(-1, 1, 2)                              # shape (N, 1, 2) required by polylines
        #cv2.polylines(img_annotated, [pts_G], isClosed=True, color=(0, 255, 0), thickness=2)
        #for (cx, cy) in exp_G:
        #   cv2.circle(img_annotated, (cx, cy), 5, (0, 255, 0), -1) # green corner dots

    if corners_R is not None:
        exp_R = expand_corners(corners_R, EXPAND_PX)
        pts_R = exp_R.reshape(-1, 1, 2)
        #cv2.polylines(img_annotated, [pts_R], isClosed=True, color=(255, 0, 0), thickness=2)
        #for (cx, cy) in exp_R:
        #    cv2.circle(img_annotated, (cx, cy), 5, (255, 0, 0), -1) # red corner dots

    # 7. Compute Intersection of the two bounding boxes
    inter_x_min = max(x_min_G, x_min_R)
    inter_y_min = max(y_min_G, y_min_R)
    inter_x_max = min(x_max_G, x_max_R)
    inter_y_max = min(y_max_G, y_max_R)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        # Boxes overlap — draw intersection region in yellow
        #cv2.rectangle(img_annotated, (inter_x_min, inter_y_min), (inter_x_max, inter_y_max), (255, 255, 0), 2)
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
    return G, R, img_annotated, bbox_G, bbox_R, bbox_I, corners_G, corners_R

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

x_min_G, y_min_G, x_max_G, y_max_G = bbox_G
x_min_R, y_min_R, x_max_R, y_max_R = bbox_R


TL_G = (x_min_G, y_min_G)
TR_G = (x_max_G, y_min_G)
BL_G = (x_min_G, y_max_G)
BR_G = (x_max_G, y_max_G)

#print(f"TL_G: {TL_G}, TR_G: {TR_G}, BL_G: {BL_G}, BR_G: {BR_G}")

TL_R = (x_min_R, y_min_R)
TR_R = (x_max_R, y_min_R)
BL_R = (x_min_R, y_max_R)
BR_R = (x_max_R, y_max_R)   

#print(f"TL_R: {TL_R}, TR_R: {TR_R}, BL_R: {BL_R}, BR_R: {BR_R}")

i = np.where(g_matrix[y_min_G, :] == 255)[0]

print(f"i: {i}")
print(f"i[0]: {i[0]}")



# Width and height of Green object:
w_G = x_max_G - x_min_G
h_G = y_max_G - y_min_G

# Width and height of Red object:
w_R = x_max_R - x_min_R
h_R = y_max_R - y_min_R

#print(f"Green Object Bounding Box: ({x_min_G}, {y_min_G}) to ({x_max_G}, {y_max_G})")
#print(f"Green Object width: {w_G}, height: {h_G}")

#print(f"Red Object Bounding Box:   ({x_min_R}, {y_min_R}) to ({x_max_R}, {y_max_R})")
#print(f"Red Object width: {w_R}, height: {h_R}")

"""
# Intersection region
if bbox_I is not None:
    x_min_I, y_min_I, x_max_I, y_max_I = bbox_I
    w_I = x_max_I - x_min_I
    h_I = y_max_I - y_min_I
    print(f"\nIntersection Region:       ({x_min_I}, {y_min_I}) to ({x_max_I}, {y_max_I})")
    print(f"Intersection width: {w_I}, height: {h_I}")
else:
    print("\nNo intersection between the two bounding boxes.")

# Exact corner coordinates
if corners_G is not None:
    print(f"\nGreen Object corners ({len(corners_G)} points):")
    for i, (cx, cy) in enumerate(corners_G):
        print(f"  Corner {i}: ({cx}, {cy})")

if corners_R is not None:
    print(f"\nRed Object corners ({len(corners_R)} points):")
    for i, (cx, cy) in enumerate(corners_R):
        print(f"  Corner {i}: ({cx}, {cy})")

"""
    


