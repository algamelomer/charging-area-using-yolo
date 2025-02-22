import requests
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

# Load YOLOv8 classification model
MODEL_PATH = "runs/classify/train7/weights/best.pt"
model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ESP32-CAM Stream URL
ESP32_STREAM_URL = "http://192.168.0.150/stream"
ESP32_CONTROL_URL = "http://192.168.0.150:81"

# Timer for stable output
last_car_time = 0  # Last time "car" was detected
COOLDOWN_SECONDS = 5  # Time to wait before turning "off" LED/relay


def send_command(device, state):
    """Send a command to the ESP32 to control LED/Relay."""
    url = f"{ESP32_CONTROL_URL}/{device}?state={state}"
    try:
        requests.get(url, timeout=2)
    except Exception as e:
        print(f"❌ Error sending command: {e}")


def classify_frame(img):
    """Classify the image using YOLOv8 model."""
    img_resized = cv2.resize(img, (224, 224))
    results = model.predict(img_resized, device=device, verbose=False)

    # Extract classification result
    probs = results[0].probs
    class_tensor = probs.data
    class_index = torch.argmax(class_tensor).item()
    confidence = class_tensor[class_index].item()

    class_labels = model.names
    return class_labels[class_index], confidence


def main():
    global last_car_time
    print("📷 Starting AI classification...")

    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    if not cap.isOpened():
        print("❌ Error: Could not open ESP32 stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read frame")
            continue  # Skip this iteration if frame is not available

        # Process the latest frame
        label, confidence = classify_frame(frame)
        current_time = time.time()

        if label.lower() == "car":
            last_car_time = current_time  # Reset cooldown timer

        # Determine if we should send "on" or "off"
        if current_time - last_car_time < COOLDOWN_SECONDS:
            send_command("led", "on")
            send_command("relay", "off")
            display_label = "Car (ON)"
            color = (0, 255, 0)
        else:
            send_command("led", "off")
            send_command("relay", "on")
            display_label = "No Car (OFF)"
            color = (0, 0, 255)

        # Draw classification result
        cv2.putText(frame, f"{display_label} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show the stream
        cv2.imshow("ESP32-CAM AI Classification", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Stopping AI classification.")


if __name__ == "__main__":
    main()
