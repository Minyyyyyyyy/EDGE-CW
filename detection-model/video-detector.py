#!/usr/bin/env python3
"""
video-detector.py

A simple script for fall detection from a video file on an edge device.
It loads a local YOLOv5 model (yolov5s.pt) for person detection and a quantized TFLite model
for fall detection classification. If a fall is detected for three consecutive seconds,
and at least three seconds have passed since the last capture, a screenshot is saved.
Additionally, the script processes every Nth frame (fast forwarding) to speed up inference.
"""

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
import time
import os
import certifi


os.environ['SSL_CERT_FILE'] = certifi.where()

# -------------------------------
# TFLite Model Loading and Inference Functions
# -------------------------------
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def classify_crop_tflite(cropped_img, interpreter, input_details, output_details, pref_size=(128, 128)):
    # Convert cropped image from BGR to RGB, resize, and normalize
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(cropped_rgb, pref_size)
    normalized = resized.astype(np.float32) / 255.0

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    if input_scale != 0:
        quantized_input = normalized / input_scale + input_zero_point
        quantized_input = np.clip(quantized_input, -128, 127).astype(np.int8)
    else:
        quantized_input = normalized

    # Add batch dimension and run inference
    input_data = np.expand_dims(quantized_input, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if output_scale != 0:
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    predicted_class = np.argmax(output_data)
    return predicted_class, output_data

# -------------------------------
# YOLOv5 Detection Model Loading (Local .pt file)
# -------------------------------
def load_yolo_model(pt_path):
    """
    Loads YOLOv5 from a local .pt file using torch.hub.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_path, force_reload=False)
    model.classes = [0]  # Restrict detections to the "person" class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

# -------------------------------
# Main Fall Detection from Video with Frame Skipping
# -------------------------------
def main():
    tflite_model_path = "Fall-detector-lite.tflite"
    yolov5_pt_path = "yolov5s.pt"
    video_path = "my-video.mp4"

    interpreter, input_details, output_details = load_tflite_model(tflite_model_path)
    yolo_model = load_yolo_model(yolov5_pt_path)
    pref_size = (128, 128)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    # Parameters for tracking fall detections per second
    fall_by_second = {}
    last_saved_second = -10  # Enforce gap between captures

    # Set skip factor
    skip_frames = 20  

    current_frame = 0

    # Create folder for saving fall captures
    capture_dir = "fall_captures"
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        # Skip frames: process only every Nth frame
        if current_frame % skip_frames != 0:
            continue

        current_second = int(current_frame / fps)

        # Process frame: Convert to RGB and then to PIL image for YOLO detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        results = yolo_model(pil_img, size=640)
        df = results.pandas().xyxy[0]

        frame_has_fall = False

        for idx, row in df.iterrows():
            if row['confidence'] > 0.5:
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                pred, _ = classify_crop_tflite(crop, interpreter, input_details, output_details, pref_size)
                if pred == 0:
                    frame_has_fall = True

                label_text = "Fall Detected" if pred == 0 else "Non-Fall"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update detection status for the current second
        fall_by_second[current_second] = fall_by_second.get(current_second, False) or frame_has_fall

        # Check if three consecutive seconds have fall detection and 3 seconds have passed since last save
        if (current_second >= 2 and 
            fall_by_second.get(current_second - 2, False) and 
            fall_by_second.get(current_second - 1, False) and 
            fall_by_second.get(current_second, False) and 
            (current_second - last_saved_second) >= 3):
            timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime())
            filename = os.path.join(capture_dir, f"fall_capture_{timestamp}_sec{current_second}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[ALERT] Fall confirmed at second {current_second}! Screenshot saved: {filename}")
            last_saved_second = current_second  # Enforce a 3-second gap before next capture

        cv2.imshow("Fall Detection (Video)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

