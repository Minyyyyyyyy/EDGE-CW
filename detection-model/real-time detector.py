#!/usr/bin/env python3
"""
fall_detection_simple_gui.py

A simple GUI for fall detection system running on Raspberry Pi 5,
using only OpenCV for display (no external GUI libraries).
This version maintains continuous live stream display while detecting falls every 3 frames.
"""

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
import os
import certifi
import time
import boto3
from dotenv import load_dotenv
import threading
import queue
from datetime import datetime

# Load environment variables
load_dotenv()

# Set SSL certificate file (for secure requests)
os.environ['SSL_CERT_FILE'] = certifi.where()

# AWS S3 Configuration
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')


class FallDetectionApp:
    def __init__(self):
        # Initialize variables
        self.tflite_model_path = "Fall-detector-lite.tflite"
        self.yolov5_pt_path = "yolov5s.pt"
        self.capture_dir = "fall_captures"
        self.is_running = False
        self.consecutive_fall_frames = 0
        self.fall_count = 0
        self.frame_queue = queue.Queue(maxsize=5)
        self.log_messages = []
        self.display_frame = None  # For storing the latest frame to display
        self.frame_lock = threading.Lock()  # Lock for thread-safe frame access

        # Window names
        self.main_window = "Fall Detection System"
        self.log_window = "Activity Log"

        # Create directory for fall captures if it doesn't exist
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)

        # Load models
        print("Loading detection models...")
        try:
            self.interpreter, self.input_details, self.output_details = self.load_tflite_model()
            self.yolo_model = self.load_yolo_model()
            print("Models loaded successfully!")
            self.log("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.log(f"Error loading models: {e}")
            raise

    def log(self, message):
        """Add message to the log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.log_messages.append(log_message)
        # Keep only the last 20 log messages
        if len(self.log_messages) > 20:
            self.log_messages.pop(0)

    def create_log_image(self):
        """Create an image containing log messages"""
        # Create a black image for the log
        log_img = np.zeros((400, 600, 3), dtype=np.uint8)

        # Add log messages
        y_pos = 30
        for msg in self.log_messages:
            cv2.putText(log_img, msg, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 20

        return log_img

    def load_tflite_model(self):
        """Load the TFLite fall detection model"""
        interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    def load_yolo_model(self):
        """Load the YOLOv5 person detection model"""
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.yolov5_pt_path, force_reload=False)
        model.classes = [0]  # Person class only
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def classify_crop_tflite(self, cropped_img, pref_size=(128, 128)):
        """Classify a person crop using the TFLite model"""
        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(cropped_rgb, pref_size)
        normalized = resized.astype(np.float32) / 255.0

        input_scale, input_zero_point = self.input_details[0]['quantization']
        output_scale, output_zero_point = self.output_details[0]['quantization']

        if input_scale != 0:
            quantized_input = normalized / input_scale + input_zero_point
            quantized_input = np.clip(quantized_input, -128, 127).astype(np.int8)
        else:
            quantized_input = normalized

        input_data = np.expand_dims(quantized_input, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        if output_scale != 0:
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        predicted_class = np.argmax(output_data)
        return predicted_class, output_data

    def upload_to_s3(self, file_path, s3_key):
        """Upload a file to S3 bucket"""
        try:
            s3 = boto3.client(
                's3',
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )

            if not os.path.exists(file_path):
                self.log(f"File '{file_path}' not found!")
                return

            # Maintain screenshot limit
            self.maintain_screenshot_limit(s3, BUCKET_NAME)

            s3.upload_file(
                Filename=file_path,
                Bucket=BUCKET_NAME,
                Key=s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            self.log(f"Uploaded '{file_path}' to S3")
        except Exception as e:
            self.log(f"Error uploading to S3: {e}")

    def maintain_screenshot_limit(self, s3, bucket, prefix="fall_frame_", max_count=10):
        """Maintain S3 screenshot count limit"""
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        except Exception as e:
            self.log(f"Error listing objects in bucket: {e}")
            return

        if "Contents" in response:
            screenshots = response["Contents"]
            if len(screenshots) >= max_count:
                screenshots.sort(key=lambda obj: obj['LastModified'])
                num_to_delete = len(screenshots) - (max_count - 1)
                self.log(f"Deleting {num_to_delete} oldest screenshots")
                for i in range(num_to_delete):
                    key_to_delete = screenshots[i]['Key']
                    try:
                        s3.delete_object(Bucket=bucket, Key=key_to_delete)
                        self.log(f"Deleted {key_to_delete}")
                    except Exception as e:
                        self.log(f"Error deleting {key_to_delete}: {e}")

    def cleanup_s3(self):
        """Clean up all screenshots in S3 bucket"""
        try:
            s3 = boto3.client(
                's3',
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )

            self.log("Starting S3 cleanup...")
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="fall_frame_")

            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj['Key']
                    try:
                        s3.delete_object(Bucket=BUCKET_NAME, Key=key)
                        self.log(f"Cleaned up {key}")
                    except Exception as e:
                        self.log(f"Error deleting {key}: {e}")
                self.log("S3 bucket cleanup completed")
            else:
                self.log("No screenshots to clean up")

        except Exception as e:
            self.log(f"Error during S3 cleanup: {e}")

    def run(self):
        """Run the application"""
        # Create the main OpenCV window
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.main_window, 800, 600)
        cv2.namedWindow(self.log_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.log_window, 600, 400)
        cv2.moveWindow(self.log_window, 820, 0)  # Position log window to the right of main window

        # Create trackbar for controls
        cv2.createTrackbar("Monitor:OFF/ON", self.main_window, 0, 1, self.toggle_monitoring)
        cv2.createTrackbar("Cleanup S3", self.main_window, 0, 1, self.trigger_cleanup)

        self.cap = None
        self.detection_thread = None
        self.log("Fall detection system ready. Toggle 'Monitor' to start.")

        # Create a default status image for when there's no camera feed
        status_img = self.create_status_image()

        # Main display loop
        while True:
            # Check if we have a frame to display
            with self.frame_lock:
                if self.display_frame is not None:
                    # Display the latest processed frame with status overlay
                    display_frame = self.add_status_overlay(self.display_frame.copy())
                    cv2.imshow(self.main_window, display_frame)
                else:
                    # Show status image if no frame is available
                    cv2.imshow(self.main_window, status_img)

            # Show log window
            log_img = self.create_log_image()
            cv2.imshow(self.log_window, log_img)

            # Process key presses
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == ord('s'):  # 's' to start/stop
                self.toggle_monitoring(not self.is_running)
            elif key == ord('c'):  # 'c' to cleanup S3
                self.cleanup_s3()
                # Reset the cleanup trackbar
                cv2.setTrackbarPos("Cleanup S3", self.main_window, 0)

        # Cleanup
        self.stop_monitoring()
        cv2.destroyAllWindows()

    def create_status_image(self):
        """Create an image showing current status when no camera feed is available"""
        # Create a dark gray background
        status_img = np.ones((480, 640, 3), dtype=np.uint8) * 64

        # Add status text
        status_text = "MONITORING ACTIVE" if self.is_running else "MONITORING STOPPED"
        status_color = (0, 255, 0) if self.is_running else (0, 0, 255)
        cv2.putText(status_img, status_text, (140, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, status_color, 2, cv2.LINE_AA)

        # Add fall count
        cv2.putText(status_img, f"Falls Detected: {self.fall_count}", (180, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # Add instructions
        cv2.putText(status_img, "Controls:", (100, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(status_img, "- Use trackbar or press 's' to start/stop", (120, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(status_img, "- Use trackbar or press 'c' to cleanup S3", (120, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(status_img, "- Press 'q' or ESC to quit", (120, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        return status_img

    def add_status_overlay(self, frame):
        """Add status overlay to video frame"""
        # Add status text in top-left corner
        status_text = "MONITORING" if self.is_running else "PAUSED"
        status_color = (0, 255, 0) if self.is_running else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, status_color, 2, cv2.LINE_AA)

        # Add fall count
        cv2.putText(frame, f"Falls: {self.fall_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def toggle_monitoring(self, value):
        """Toggle monitoring on/off"""
        if isinstance(value, int):
            # Called from trackbar
            should_run = bool(value)
        else:
            # Called directly with a boolean
            should_run = value
            # Update trackbar to match
            cv2.setTrackbarPos("Monitor:OFF/ON", self.main_window, 1 if should_run else 0)

        if should_run and not self.is_running:
            self.start_monitoring()
        elif not should_run and self.is_running:
            self.stop_monitoring()

    def trigger_cleanup(self, value):
        """Trigger S3 cleanup when trackbar is moved to 1"""
        if value == 1:
            self.cleanup_s3()
            # Reset the trackbar to 0
            cv2.setTrackbarPos("Cleanup S3", self.main_window, 0)

    def start_monitoring(self):
        """Start the fall detection monitoring"""
        if self.is_running:
            return

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Unable to open camera.")

            self.is_running = True
            self.consecutive_fall_frames = 0
            self.log("Started fall detection monitoring")

            # Start the video capture thread for continuous feed
            self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.video_thread.start()

            # Start the detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()

        except Exception as e:
            self.log(f"Error starting monitoring: {e}")
            self.stop_monitoring()

    def video_capture_loop(self):
        """Continuously capture video frames for display"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.log("Error: Unable to read frame from camera")
                    time.sleep(0.1)
                    continue

                # Update the display frame with the new frame
                with self.frame_lock:
                    self.display_frame = frame.copy()

                # Brief sleep to avoid maxing out CPU
                time.sleep(0.01)

            except Exception as e:
                self.log(f"Error in video capture loop: {e}")
                time.sleep(0.1)

    def detection_loop(self):
        """Fall detection processing loop running in a separate thread"""
        # Frame counter for processing every Nth frame
        frame_counter = 0

        while self.is_running:
            try:
                # Get the current frame for processing
                with self.frame_lock:
                    if self.display_frame is None:
                        time.sleep(0.01)
                        continue

                    # Make a copy of the current frame for processing
                    process_frame = self.display_frame.copy()

                # Only process every 3rd frame to save resources
                frame_counter += 1
                if frame_counter % 3 != 0:
                    time.sleep(0.01)  # Small delay when not processing
                    continue

                # Convert frame for detection
                frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Run person detection
                results = self.yolo_model(pil_img, size=640)
                df = results.pandas().xyxy[0]  # Get detection results as dataframe
                fall_detected_in_frame = False

                # Process detection results
                for idx, row in df.iterrows():
                    if row['confidence'] > 0.5 and row['class'] == 0:  # Ensure person class (class 0)
                        x1, y1 = int(row['xmin']), int(row['ymin'])
                        x2, y2 = int(row['xmax']), int(row['ymax'])

                        # Ensure crop coordinates are valid
                        if y1 >= y2 or x1 >= x2 or y1 < 0 or x1 < 0 or y2 > process_frame.shape[0] or x2 > \
                                process_frame.shape[1]:
                            continue

                        crop = process_frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        # Classify the crop (person fall detection)
                        pred, raw_output = self.classify_crop_tflite(crop)

                        # Update display frame with bounding box for detected person only
                        with self.frame_lock:
                            if self.display_frame is not None:
                                # If fall detected
                                if pred == 0:  # Fall detected
                                    fall_detected_in_frame = True
                                    cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(self.display_frame, "FALL DETECTED", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                else:
                                    cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(self.display_frame, "NO FALL DETECTED", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                # Update fall detection counter
                if fall_detected_in_frame:
                    self.consecutive_fall_frames += 1
                else:
                    self.consecutive_fall_frames = 0

                # Handle confirmed fall detection (3 consecutive frames)
                if self.consecutive_fall_frames >= 3:
                    self.fall_count += 1
                    self.log("ALERT: Fall detected!")

                    # Save a screenshot of the fall
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                    save_path = os.path.join(self.capture_dir, f"fall_frame_{timestamp}.jpg")

                    with self.frame_lock:
                        if self.display_frame is not None:
                            cv2.imwrite(save_path, self.display_frame)

                    self.log(f"Fall screenshot saved: {save_path}")

                    # Upload in a separate thread to not block detection
                    s3_key = f"fall_frame_{timestamp}.jpg"
                    upload_thread = threading.Thread(
                        target=self.upload_to_s3,
                        args=(save_path, s3_key),
                        daemon=True
                    )
                    upload_thread.start()

                    # Reset consecutive fall frames but don't reset the count
                    self.consecutive_fall_frames = 0

            except Exception as e:
                self.log(f"Error in detection loop: {e}")
                time.sleep(0.1)  # Small delay to prevent tight loop on errors


    def stop_monitoring(self):
        """Stop the fall detection monitoring"""
        self.is_running = False

        # Wait for threads to finish
        for thread_name in ['video_thread', 'detection_thread']:
            if hasattr(self, thread_name) and getattr(self, thread_name) is not None:
                thread = getattr(self, thread_name)
                if thread.is_alive():
                    thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Clear the display frame
        with self.frame_lock:
            self.display_frame = None

        self.log("Stopped fall detection monitoring")


if __name__ == "__main__":
    app = FallDetectionApp()
    app.run()
