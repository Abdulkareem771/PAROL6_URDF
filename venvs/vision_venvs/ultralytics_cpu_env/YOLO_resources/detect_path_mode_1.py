import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ==================== CONFIGURATION ====================

# Path to current directory
current_dir = Path(__file__)

# Path to project directory
project_dir = current_dir.parent.parent

# Path to an official model
MODEL_PATH_OFFICIAL = project_dir / "yolov8n-seg.pt"

# Path to your trained YOLO model
MODEL_PATH = project_dir / "yolo_training" / "experiment_12" / "weights" / "best.pt"   # replace with your path

# Path to the folder containing images
IMAGE_FOLDER = project_dir / "data" / "raw_images_ROI_model"    # replace with your folder path

# Path to single image
SINGLE_IMAGE_PATH = project_dir / "data" / "YOLO_Segmentation_data" / "test" / "26.jpg"

# Padding around bounding boxes (in pixels) - useful to include some context
PADDING = 20  # Set to 0 for no padding


# Load a model
model = YOLO(MODEL_PATH_OFFICIAL)  # load an official model
model = YOLO(MODEL_PATH)  # load a custom model

# Predict with the model
results = model(SINGLE_IMAGE_PATH)  # predict on an image

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)