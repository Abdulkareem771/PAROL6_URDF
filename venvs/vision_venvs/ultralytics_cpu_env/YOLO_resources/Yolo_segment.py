import cv2
import numpy as np
import os
import random
import sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# --- Configuration ---
# Adjust these paths to match your project structure
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "YOLO_Segmentation_data"
OUTPUT_DIR = BASE_DIR / "data" / "YOLO_Segmentation_results"

SINGLE_IMAGE = BASE_DIR / "data" / "YOLO_Segmentation_data" / "test" / "26.jpg"

# Model paths
MODEL_PATH = BASE_DIR / "yolo_segmentation_models_results" / "experiment_2" / "weights" / "best.pt"




# 1. Read the image
img = cv2.imread(str(SINGLE_IMAGE))
if img is None:
    print(f"Could not read image: {SINGLE_IMAGE}")
    exit()

# Convert BGR to RGB for correct display in Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_annotated = img_rgb.copy()
# Load a trained best Segment model
model = YOLO(MODEL_PATH)

# Run inference on an image
results = model(SINGLE_IMAGE)  # results list



#print(f"results: {results}")




for r in results:
    boxes = r.boxes  # Boxes object for detected boxes
    masks = r.masks.numpy()  # Masks object for segmentation masks
   #print(f"boxes: {boxes}")
    #print(f"masks[0]: {masks[0]}")

#print(f"masks.data: {masks.data}")
#print(f"masks.data.shape: {(masks.data)[0]}")

#print(f"masks.xy: {masks.xy}")
print(f"number of masks: {len(masks.xy)}")

print(f"masks.xy[0]: {masks.xy[0]}")
print(f"masks.xy[1]: {masks.xy[1]}")

object_mask = masks.xy()
g_mask = object_mask[0]
b_mask = object_mask[1]

print(f"g_mask: {g_mask}")
print(f"b_mask: {b_mask}")


G = (masks.data)[0]
B = (masks.data)[1]

# View results
#for r in results:
#    print(r.masks.numpy())  # print the Masks object containing the detected instance masks

"""

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
   

"""


"""
# 6. GUI Display Section
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
plt.title("B Matrix (Blue Object)")
plt.imshow(B, cmap='gray')
plt.axis('off')

# Subplot 4: Annotated Image with Bounding Boxes
plt.subplot(1, 4, 4)
plt.title("Seam Path")
plt.imshow(img_annotated)
plt.axis('off')

plt.tight_layout()
plt.show()

"""