from ultralytics import YOLO

# Load a pre-trained YOLO detection model
model = YOLO('yolo11n.pt')

# --------------------------
# Train the YOLO model
# --------------------------
train_results = model.train(
    data='path/to/yolodataset/data.yaml',
    epochs=100,
    imgsz=640
)