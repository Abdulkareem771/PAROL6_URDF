from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(
    data="dataset/data.yaml",
    epochs=70,
    imgsz=640,
    batch=2,
    workers=2,
    device='cpu',
    project='yolo_training',
    name='experiment_1',
    patience=5,

)




