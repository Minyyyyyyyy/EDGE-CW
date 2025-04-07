#!/usr/bin/env python3
"""
real-time detector.py

A simple script for real-time fall detection on an edge device (e.g. Raspberry Pi).
It loads a local YOLOv5 model (yolov5s.pt) for person detection and a quantized TFLite model
for fall detection classification. If a fall (class 0) is detected in three consecutive frames,
it saves the cropped image of the fallen person.
"""

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
import os
import certifi
import time

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
    # Convert cropped image from BGR to RGB, resize, and normalize the image
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(cropped_rgb, pref_size)
    normalized = resized.astype(np.float32) / 255.0

    # Get quantization parameters for input and output
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
    # Load local custom model 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_path, force_reload=False)
    model.classes = [0]  # Restrict detections to the "person" class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

# -------------------------------
# Main Real-Time Detection Loop
# -------------------------------
def main():
    tflite_model_path = "Fall-detector-lite.tflite"
    yolov5_pt_path = "yolov5s.pt"

    interpreter, input_details, output_details = load_tflite_model(tflite_model_path)
    pref_size = (128, 128)

    yolo_model = load_yolo_model(yolov5_pt_path)

    # Create folder to save fall capture images
    capture_dir = "fall_captures"
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    print("Starting real-time fall detection. Press 'q' to exit.")

    # Counter for consecutive frames with fall detection
    consecutive_fall_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        # Convert frame to RGB and then to PIL image (required for YOLO)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Run YOLOv5 detection; using size=640 for detection
        results = yolo_model(pil_img, size=640)
        df = results.pandas().xyxy[0]
        fall_detected_in_frame = False
        fall_crops = []  # To store crops that are detected as fall

        for idx, row in df.iterrows():
            if row['confidence'] > 0.5:
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Classify the cropped region using the TFLite model
                pred, raw_output = classify_crop_tflite(crop, interpreter, input_details, output_details, pref_size)
                print(f"Raw output: {raw_output}, Predicted class: {pred}")

                # If predicted class 0 (Fall Detected), mark frame as fall and store crop
                if pred == 0:
                    fall_detected_in_frame = True
                    fall_crops.append(crop)
                    label_text = "Fall Detected"
                else:
                    label_text = "Non-Fall"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Check if the current frame had any fall detections
        if fall_detected_in_frame:
            consecutive_fall_frames += 1
        else:
            consecutive_fall_frames = 0

        # If 3 consecutive frames have fall detections, save the crops from the current frame
        if consecutive_fall_frames >= 3:
            timestamp = time.localtime()
            formatted_time = time.strftime("%m-%d %H:%M:%S", timestamp)
            for i, crop in enumerate(fall_crops):
                save_path = os.path.join(capture_dir, f"fall_{formatted_time}_{i}.jpg")
                cv2.imwrite(save_path, crop)
                print(f"Saved fall capture: {save_path}")
            # Reset counter to avoid repeated captures for the same event
            consecutive_fall_frames = 0

        if fall_detected_in_frame:
            print("Fall detected in current frame!")
        
        cv2.imshow("Real-Time Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


