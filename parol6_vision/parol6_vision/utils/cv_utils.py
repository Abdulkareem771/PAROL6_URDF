"""
cv_utils.py

Computer vision utility functions for image processing and transformation.
"""

import cv2
import numpy as np

def normalize_image(image):
    """Normalize image to 0-1 range."""
    return image.astype(np.float32) / 255.0

def contrast_stretch(image):
    """Apply contrast stretching."""
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = np.interp(image, (p2, p98), (0, 255)).astype(np.uint8)
    return img_rescale
