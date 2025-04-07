#!/usr/bin/env python3
"""
app.py

A simple script for real-time fall detection on an edge device (e.g. Raspberry Pi).
It loads a local YOLOv5 model (yolov5s.pt) for person detection and a quantized TFLite model
for fall detection classification.
"""

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
import os
import certifi

# Set SSL certificate file (if needed)
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
    # Convert BGR to RGB, resize and normalize the image
    cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(cropped_rgb, pref_size)
    normalized = resized.astype(np.float32) / 255.0

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    # Quantize input if needed
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
    # Use 'custom' to load your local weights.
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_path, force_reload=False)
    model.classes = [0]  # Restrict detections to the person class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

# -------------------------------
# Main Real-Time Detection Loop
# -------------------------------
def main():
    # Path to your local TFLite and YOLOv5 model files
    tflite_model_path = "/Users/Muaadh Nazly/Muaadh/IIT/Year 3 AI & DS/Semester 2/CM 3603 Edge AI/CW/Implementation/Github/EDGE-CW/Fall-detector-lite.tflite" 
    yolov5_pt_path = "/Users/Muaadh Nazly/Muaadh/IIT/Year 3 AI & DS/Semester 2/CM 3603 Edge AI/CW/Implementation/Github/EDGE-CW/yolov5s.pt"  

    # Load TFLite model for fall detection
    interpreter, input_details, output_details = load_tflite_model(tflite_model_path)
    pref_size = (128, 128)

    # Load YOLOv5 model for person detection from the local .pt file
    yolo_model = load_yolo_model(yolov5_pt_path)

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    print("Starting real-time fall detection. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        # Convert frame to RGB and to a PIL image for YOLO processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Run YOLOv5 detection; using size=640 for detection (adjust as needed)
        results = yolo_model(pil_img, size=640)
        df = results.pandas().xyxy[0]
        fall_detected = False

        # Process detections with confidence > 0.5
        for idx, row in df.iterrows():
            if row['confidence'] > 0.5:
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                pred, raw_output = classify_crop_tflite(crop, interpreter, input_details, output_details, pref_size)
                print(f"Raw output: {raw_output}, Predicted class: {pred}")

                if pred == 0:
                    label_text = "Fall Detected"
                    fall_detected = True
                else:
                    label_text = "Non-Fall"

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if fall_detected:
            print("Fall detected in current frame!")

        cv2.imshow("Real-Time Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
