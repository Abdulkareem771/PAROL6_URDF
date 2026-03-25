import cv2
import numpy as np
import os
import glob
from pathlib import Path


CEXPAND_PX    = 12  # pixels to dilate each contour mask outward

current_dir = Path(__file__)                # YOLO_resources/detect_path.py
project_dir = current_dir.parent.parent     # ultralytics_cpu_env

SINGLE_IMAGE = project_dir / "data" / "some_images" / "image_2.png"

def segment_blocks(image_path):
    # 1. Read the image
    img = cv2.imread(str(image_path))
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
    
    # Blue range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])


    # 4. Create Masks (G and R matrices)
    # G matrix: 255 for green pixels, 0 otherwise
    G = cv2.inRange(hsv, lower_green, upper_green)

    # B matrix: 255 for blue pixels, 0 otherwise
    B = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Optional: Clean up noise with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    G = cv2.morphologyEx(G, cv2.MORPH_OPEN, kernel)
    B = cv2.morphologyEx(B, cv2.MORPH_OPEN, kernel)

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
    contour_B = find_contours(B)
    """
    if contour_G is not None:
        cv2.drawContours(img_annotated, [contour_G], -1, (255, 0, 0), 4)   # red outline (green object)

    if contour_B is not None:
        cv2.drawContours(img_annotated, [contour_B], -1, (255, 0, 0), 4)   # red outline (red object)
    """
    # Expand contours outward by CEXPAND_PX using morphological dilation
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*CEXPAND_PX+1, 2*CEXPAND_PX+1))
    G_exp = cv2.dilate(G, dil_kernel)
    B_exp = cv2.dilate(B, dil_kernel)

    contour_G_exp = find_contours(G_exp)
    contour_B_exp = find_contours(B_exp)
    """
    if contour_G_exp is not None:
        cv2.drawContours(img_annotated, [contour_G_exp], -1, (255, 0, 0), 4)  # red = expanded green contour

    if contour_B_exp is not None:
        cv2.drawContours(img_annotated, [contour_B_exp], -1, (255, 0, 0), 4)  # red = expanded red contour
    """
    # Intersection of the two expanded contour regions
    intersection_mask = cv2.bitwise_and(G_exp, B_exp)
    contour_I = find_contours(intersection_mask)
    
    if contour_I is not None:
        cv2.drawContours(img_annotated, [contour_I], -1, (255, 0, 0), -1)    # red = intersection region
    
    return img_annotated


img_annotated = segment_blocks(SINGLE_IMAGE)

if img_annotated is not None:
    cv2.imshow("Seam Path", img_annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
