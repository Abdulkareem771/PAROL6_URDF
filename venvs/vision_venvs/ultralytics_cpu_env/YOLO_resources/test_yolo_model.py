from ultralytics import YOLO
import os
from pathlib import Path

current_dir = Path(__file__)
project_dir = current_dir.parent.parent

print(f"current_dir={current_dir}")
print(f"project_dir={project_dir}")


# Path to your trained model
MODEL_PATH = project_dir / "yolo_training" / "experiment_9" / "weights" / "best.pt"   # replace with your path

# Path to the folder containing images to label
IMAGE_FOLDER = project_dir / "data" / "Images_to_test"    # replace with your folder path

# Path to the folder containing the labels files and images
RESULTS_FOLDER = project_dir / "data" / "Models_testing" / "test_results_experiment_9"    # replace with your folder path
#RESULTS_FOLDER = project_dir / "data" / "Models_testing"


# Confidence threshold for detection
CONF_THRESHOLD = 0.5

# Path to single image to be labeled
SINGLE_IMAGE_PATH = project_dir / "data" / "Images_to_test" / "20260219_224015.jpg"    # replace with your folder path



def detect_work_pieces(input_dir, output_dir = RESULTS_FOLDER ):
                    
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run detection
    results = model.predict(
        source=input_dir,
        conf=CONF_THRESHOLD,
        save=True,
        project=output_dir,
        name='labeling_images',
        exist_ok=True,
        #save_txt=True,
        #save_conf=True,
        #stream=True
    )
    
    print(f"Detection complete! Results saved to: {output_dir}")
    return results

# Usage:

#detect_work_pieces(IMAGE_FOLDER)

detect_work_pieces(SINGLE_IMAGE_PATH)
