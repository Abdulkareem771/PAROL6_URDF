from ultralytics import YOLO
import os
from pathlib import Path

current_dir = Path(__file__).parent
project_dir = current_dir.parent.parent

print(f"current_dir={current_dir}")
print(f"project_dir={project_dir}")

# Path to your trained model
MODEL_PATH = project_dir / "YOLO" / "yolo_training" / "experiment_1" / "weights" / "best.pt"   # replace with your path

# Path to the folder containing images to label
IMAGE_FOLDER = project_dir / "src" / "data" / "image to label"     # replace with your folder path

# Path to the folder containing the labels files and images
RESULTS_FOLDER = project_dir / "results"     # replace with your folder path

# Confidence threshold for detection
CONF_THRESHOLD = 0.5 



def detect_whiteboards(input_dir, output_dir = RESULTS_FOLDER ):
                    
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
        save_txt=True,
        #save_conf=True,
        #stream=True
    )
    
    print(f"Detection complete! Results saved to: {output_dir}")
    return results

# Usage
detect_whiteboards(IMAGE_FOLDER)
