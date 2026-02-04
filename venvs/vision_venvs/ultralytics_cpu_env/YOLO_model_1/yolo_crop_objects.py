"""
YOLO Object Detection and Cropping Script
==========================================
This script detects objects in images using a trained YOLO11n model,
crops the detected objects, and saves them to a separate folder.

Author: Auto-generated
Date: 2026-02-04
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np

current_dir = Path(__file__).parent
project_dir = current_dir.parent


#print(f"current_dir={current_dir}")
#print(f"project_dir={project_dir}")

# ==================== CONFIGURATION ====================

# Path to your trained YOLO model
MODEL_PATH = project_dir / "YOLO_model_1" / "yolo_training" / "experiment_1" / "weights" / "best.pt"

# Confidence threshold for detection (adjust based on your F1 curve)
CONF_THRESHOLD = 0.25

# IoU threshold for Non-Maximum Suppression
IOU_THRESHOLD = 0.7

# Output folder for cropped images
OUTPUT_FOLDER = project_dir / "data" / "ROI_images"

# Padding around bounding boxes (in pixels) - useful to include some context
PADDING = 10  # Set to 0 for no padding


# ==================== MAIN FUNCTIONS ====================

def detect_and_crop_objects(image_path, model, output_dir, conf_threshold=CONF_THRESHOLD, 
                           iou_threshold=IOU_THRESHOLD, padding=PADDING):
    """
    Detect objects in an image and crop them to separate files.
    
    Args:
        image_path (str): Path to the input image
        model: Loaded YOLO model
        output_dir (str): Directory to save cropped images
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for NMS
        padding (int): Padding in pixels around bounding boxes
        
    Returns:
        int: Number of objects detected and cropped
    """
    # Read the original image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return 0
    
    img_height, img_width = img.shape[:2]
    
    # Run YOLO detection
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False  # Set to True to see detection details
    )
    
    # Get the first result (single image)
    result = results[0]
    
    # Get detection boxes
    boxes = result.boxes
    num_detections = len(boxes)
    
    if num_detections == 0:
        print(f"No objects detected in {Path(image_path).name}")
        return 0
    
    # Get image name without extension
    image_name = Path(image_path).stem
    
    # Process each detection
    cropped_count = 0
    for idx, box in enumerate(boxes):
        # Get bounding box coordinates (xyxy format)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Add padding
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(img_width, x2 + padding)
        y2_padded = min(img_height, y2 + padding)
        
        # Get class ID and confidence
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        
        # Crop the object
        cropped_img = img[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Create output filename
        output_filename = f"{image_name}_obj{idx+1}_{class_name}_conf{confidence:.2f}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the cropped image
        cv2.imwrite(output_path, cropped_img)
        cropped_count += 1
        
        print(f"  → Saved: {output_filename} | Class: {class_name} | Conf: {confidence:.2f}")
    
    return cropped_count


def process_single_image(image_path, model_path=MODEL_PATH, output_folder=OUTPUT_FOLDER,
                        conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD, 
                        padding=PADDING):
    """
    Process a single image: detect objects and save cropped regions.
    
    Args:
        image_path (str): Path to input image
        model_path (str): Path to YOLO model weights
        output_folder (str): Output directory for cropped images
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
        padding (int): Padding around bounding boxes
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Process the image
    print(f"\nProcessing image: {image_path}")
    num_cropped = detect_and_crop_objects(
        image_path, model, output_folder, 
        conf_threshold, iou_threshold, padding
    )
    
    print(f"\n✓ Processing complete!")
    print(f"  Total objects detected and cropped: {num_cropped}")
    print(f"  Cropped images saved to: {output_folder}/")


def process_image_folder(input_folder, model_path=MODEL_PATH, output_folder=OUTPUT_FOLDER,
                        conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                        padding=PADDING):
    """
    Process all images in a folder: detect objects and save cropped regions.
    
    Args:
        input_folder (str): Path to folder containing images
        model_path (str): Path to YOLO model weights
        output_folder (str): Output directory for cropped images
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
        padding (int): Padding around bounding boxes
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Process each image
    total_cropped = 0
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        num_cropped = detect_and_crop_objects(
            str(img_path), model, output_folder,
            conf_threshold, iou_threshold, padding
        )
        total_cropped += num_cropped
    
    print(f"\n✓ Batch processing complete!")
    print(f"  Total images processed: {len(image_files)}")
    print(f"  Total objects cropped: {total_cropped}")
    print(f"  Cropped images saved to: {output_folder}/")


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    # Example 1: Process a single image
    # Uncomment and modify the path below to use
    """
    process_single_image(
        image_path= project_dir / "data" / "raw_model2_data" / "1.png",
        output_folder= project_dir / "data" /"ROI_images"
    )
    """
    
    # Example 2: Process all images in a folder
    # Uncomment and modify the path below to use
    
    process_image_folder(
        input_folder= project_dir / "data" / "dataset_model_1" / "images" / "test",
        output_folder= project_dir / "data" /"ROI_images"
    )
    
    
    # Example 3: Process with custom parameters
    """
    process_single_image(
        image_path="test_image.jpg",
        model_path="/path/to/your/model/best.pt",
        output_folder="cropped_objects",
        conf_threshold=0.5,  # Higher confidence
        iou_threshold=0.7,    # NMS threshold
        padding=20            # More padding around objects
    )
    """
    
    print("=" * 60)
    print("YOLO Object Detection and Cropping Script")
    print("=" * 60)
    print("\nPlease uncomment one of the usage examples above and")
    print("modify the paths to match your setup.")
    print("\nAvailable functions:")
    print("  1. process_single_image() - Process one image")
    print("  2. process_image_folder() - Process all images in a folder")
    print("\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    print(f"  Padding: {PADDING}px")
    print("=" * 60)
