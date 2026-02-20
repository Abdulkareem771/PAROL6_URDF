"""
Quick Start Example - YOLO Object Cropping
===========================================
Simple example demonstrating how to use the yolo_crop_objects script.
"""

from yolo_crop_objects import process_single_image, process_image_folder
import os
from pathlib import Path

current_dir = Path(__file__).parent
project_dir = current_dir.parent


#print(f"current_dir={current_dir}")
#print(f"project_dir={project_dir}")


# Path to your trained model
MODEL_PATH = project_dir / "yolo_training" / "experiment_9" / "weights" / "best.pt"   # replace with your path

# Path to the folder containing images to label
IMAGE_FOLDER = project_dir / "data" / "Images_to_test"    # replace with your folder path

# Path to the folder containing the cropped images "ROI"
RESULTS_FOLDER = project_dir / "data" / "ROI_images"
#RESULTS_FOLDER = project_dir / "data" / "ROI_images"

# Path to single image
SINGLE_IMAGE_PATH = project_dir / "data" / "Images_to_test" / "20260219_224015.jpg"

# ==================== QUICK START EXAMPLES ====================

# Example 1: Process a single image
def example_single_image():
    """Process a single image and crop detected objects."""
    process_single_image(
        image_path= SINGLE_IMAGE_PATH,  # Change this to your image path
        output_folder= RESULTS_FOLDER
    )


# Example 2: Process all images in a folder
def example_batch_processing():
    """Process all images in a folder."""
    process_image_folder(
        input_folder=IMAGE_FOLDER,  # Change this to your images folder
        output_folder=RESULTS_FOLDER
    )


# Example 3: Process with custom settings
def example_custom_settings():
    """Process with custom confidence threshold and padding."""
    process_single_image(
        image_path="path/to/your/image.jpg",
        output_folder="ROI_images",
        conf_threshold=0.5,  # Higher confidence (fewer detections, more accurate)
        padding=20           # More padding around objects
    )


# ==================== RUN ====================

if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    #example_single_image()
    # example_batch_processing()
    # example_custom_settings()
    
    print("Please uncomment one of the examples above to run.")
