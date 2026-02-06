from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(
    data="data/dataset_model_1_v2/data.yaml",
    epochs=100,
    imgsz=640,
    batch=3,
    workers=2,
    device='cpu',
    project='yolo_training',
    name='experiment_2',
    patience=5,

)




