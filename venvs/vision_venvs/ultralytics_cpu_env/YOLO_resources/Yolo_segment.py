import cv2
import numpy as np
import os
import random
import sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# --- Configuration ---
# Adjust these paths to match your project structure
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "YOLO_Segmentation_data"
OUTPUT_DIR = BASE_DIR / "data" / "YOLO_Segmentation_results"

# Model paths
MODEL_PATH = BASE_DIR / "yolo_segmentation_models_results" / "experiment_2" / "weights" / "best.pt"



# Load a trained best Segment model
model = YOLO(MODEL_PATH)

# Run inference on an image
results = model(DATA_DIR / "test" / "26.jpg")  # results list


"""
for r in results:
    boxes = r.boxes  # Boxes object for detected boxes
    masks = r.masks  # Masks object for segmentation masks
    probs = r.probs  # Probs object for classification probabilities
    print(f"boxes: {boxes}")
    print(f"masks: {masks}")
    print(f"probs: {probs}")


"""

# View results
#for r in results:
#    print(r.masks.numpy())  # print the Masks object containing the detected instance masks


# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
   


