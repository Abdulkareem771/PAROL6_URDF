from ultralytics import YOLO
import os
from pathlib import Path

current_dir = Path(__file__)
project_dir = current_dir.parent.parent

#print(f"current_dir={current_dir}")
#print(f"project_dir={project_dir}")


# Path to your Pre-trained model
MODEL_PATH = project_dir / "yolo11n.pt"

# Path to the dataset folder
DATASET_FOLDER = project_dir / "data" / "dataset_model_1_v10" / "data.yaml"    # replace with your folder path

# Path to the results folder
RESULTS_FOLDER = project_dir / "yolo_training"   # replace with your folder path



# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(
    data=DATASET_FOLDER,
    epochs=100,
    imgsz=640,
    batch=2,
    workers=4,
    device='cpu',
    project=RESULTS_FOLDER,
    name='experiment_10',
    patience=5,

)



