from ultralytics import YOLO

# Load the YOLOv8 classification model
model = YOLO("yolo11n-cls.pt")

# Train the model on your dataset
if __name__ == "__main__":
    model.train(
        data="dataset",
        epochs=350,
        imgsz=224,
        batch=32,
        weight_decay=0.0005,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mixup=0.2,
        # patience=10  # Early stopping
    )