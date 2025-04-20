#!/usr/bin/env python3
"""
fall_detection_dual_mode_gui.py

A GUI for fall detection system running on Raspberry Pi 5,
using OpenCV for display (no external GUI libraries).
Supports both image-based and sound-based fall detection methods.
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
import sounddevice as sd
from scipy.signal import spectrogram
from pydub import AudioSegment
from scipy.io.wavfile import write

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
        self.sound_model_path = "best_model3.keras"
        self.capture_dir = "../fall_captures"
        self.is_running = False
        self.consecutive_fall_frames = 0
        self.fall_count = 0
        self.frame_queue = queue.Queue(maxsize=5)
        self.log_messages = []
        self.display_frame = None  # For storing the latest frame to display
        self.frame_lock = threading.Lock()  # Lock for thread-safe frame access

        # Detection mode (0 = image, 1 = sound)
        self.detection_mode = 0

        # Sound detection settings
        self.audio_duration = 3  # seconds
        self.sample_rate = 16000
        self.spec_resize_shape = (64, 64)
        self.audio_fall_confidence = 0.0

        # Window names
        self.main_window = "Fall Detection System"
        self.log_window = "Activity Log"

        # Create directory for fall captures if it doesn't exist
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)

        # Load models
        print("Loading detection models...")
        try:
            # Load image-based models
            self.interpreter, self.input_details, self.output_details = self.load_tflite_model()
            self.yolo_model = self.load_yolo_model()

            # Load sound-based model
            self.sound_model = self.load_sound_model()

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
        """Create an improved log display"""
        # Create a dark background with gradient
        log_img = np.zeros((400, 600, 3), dtype=np.uint8)
        for i in range(400):
            alpha = i / 400.0
            color = (int(40 * alpha), int(40 * alpha), int(50 * alpha))
            log_img[i, :] = color

        # Add header
        cv2.rectangle(log_img, (0, 0), (600, 40), (40, 40, 60), -1)
        cv2.putText(log_img, "Activity Log", (220, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (200, 200, 255), 2, cv2.LINE_AA)

        # Create log panel
        cv2.rectangle(log_img, (10, 50), (590, 390), (30, 30, 40), -1)
        cv2.rectangle(log_img, (10, 50), (590, 390), (60, 60, 80), 1)

        # Add log messages with improved styling
        if not self.log_messages:
            cv2.putText(log_img, "No activity recorded yet", (150, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 180), 1, cv2.LINE_AA)
        else:
            y_pos = 80
            for i, msg in enumerate(self.log_messages):
                # Extract timestamp from log message
                if '[' in msg and ']' in msg:
                    time_part = msg[:msg.find(']') + 1]
                    content_part = msg[msg.find(']') + 1:]

                    # Draw timestamp in different color
                    cv2.putText(log_img, time_part, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (180, 180, 220), 1, cv2.LINE_AA)

                    # Draw message content
                    msg_color = (255, 200, 200) if "ALERT" in content_part else (200, 200, 200)
                    cv2.putText(log_img, content_part, (120, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, msg_color, 1, cv2.LINE_AA)
                else:
                    # Fallback for messages without timestamp format
                    cv2.putText(log_img, msg, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (200, 200, 200), 1, cv2.LINE_AA)

                y_pos += 16

                # Add separator line between messages
                if i < len(self.log_messages) - 1:
                    cv2.line(log_img, (20, y_pos - 8), (580, y_pos - 8), (60, 60, 80), 1)

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

    def load_sound_model(self):
        """Load the sound-based fall detection model"""
        model = tf.keras.models.load_model(self.sound_model_path)
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

    def record_audio_in_memory(self):
        """Record audio using sounddevice with a specified input device"""
        device_index = 0  # Device index for the Logitech Brio webcam microphone

        # Set the default device to the webcam microphone explicitly
        sd.default.device = (device_index, None)  # Set only the input device, not output

        # Print current default device to verify
        print(f"Current default input device: {sd.default.device}")

        # Record audio from the specified device
        audio = sd.rec(int(self.audio_duration * self.sample_rate),
                       samplerate=self.sample_rate, channels=1, dtype='int16', device=device_index)
        sd.wait()  # Wait for the recording to finish

        # Print current default device after recording to verify if it's still set correctly
        print(f"Current default input device after recording: {sd.default.device}")

        return audio.squeeze()  # Remove extra dimension

    def process_audio_array(self, audio):
        """Process audio array to get spectrogram for model input"""
        if audio.ndim > 1:
            audio = audio[:, 0]
        freqs, times, spec = spectrogram(audio, fs=self.sample_rate)
        log_spec = 10 * np.log10(spec + 1e-10)
        resized = cv2.resize(log_spec, self.spec_resize_shape)
        normalized = (resized - resized.min()) / (resized.max() - resized.min())
        return normalized[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    def predict_fall_from_audio(self):
        """Predict fall from audio using the sound model"""
        try:
            # Record audio
            audio = self.record_audio_in_memory()

            # Get the spectrogram
            spec_input = self.process_audio_array(audio)  # shape: (1, 64, 64, 1)

            # Predict using the spectrogram input
            prediction = self.sound_model.predict(spec_input, verbose=0)[0][0]

            # Store confidence for display
            self.audio_fall_confidence = prediction

            # Log the confidence level
            self.log(f"Sound analysis confidence: {prediction:.4f}")

            # Return True if fall detected (confidence > 0.5)
            return prediction > 0.32
        except Exception as e:
            self.log(f"Error in audio prediction: {e}")
            return False

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
        """Clean up all screenshots and audio files in S3 bucket"""
        try:
            s3 = boto3.client(
                's3',
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )

            self.log("Starting S3 cleanup...")

            # List all objects in the S3 bucket that have the fall_frame_ or fall_sound_ prefix
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="fall_")

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
                self.log("No fall-related files to clean up")

        except Exception as e:
            self.log(f"Error during S3 cleanup: {e}")

    def change_detection_mode(self, value):
        """Change the detection mode when trackbar is moved"""
        self.detection_mode = value
        mode_name = "Image Detection" if value == 0 else "Sound Detection"
        self.log(f"Detection mode changed to: {mode_name}")

        # If monitoring is already running, restart it with the new mode
        if self.is_running:
            self.stop_monitoring()
            self.start_monitoring()

    def run(self):
        """Run the application with improved window management"""
        # Create the main OpenCV window with better layout
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.main_window, 800, 600)

        # Create log window and position it to the right
        cv2.namedWindow(self.log_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.log_window, 600, 400)
        cv2.moveWindow(self.log_window, 820, 0)

        # Set window properties for more modern look (if available in OpenCV version)
        try:
            # These may not work on all OpenCV versions
            cv2.setWindowProperty(self.main_window, cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty(self.log_window, cv2.WND_PROP_TOPMOST, 1)
        except:
            pass

        # Create more visually appealing trackbars
        # Using custom images for trackbars isn't supported in OpenCV, but we can improve the labels
        cv2.createTrackbar("Monitor: OFF/ON", self.main_window, 0, 1, self.toggle_monitoring)
        cv2.createTrackbar("Mode: Image/Sound", self.main_window, 0, 1, self.change_detection_mode)
        cv2.createTrackbar("Clean S3 Storage", self.main_window, 0, 1, self.trigger_cleanup)

        self.cap = None
        self.detection_thread = None
        self.log("Fall detection system ready")
        self.log("Toggle 'Monitor' to begin detection")

        # Create initial status images
        status_img = self.create_status_image()

        # Main display loop with better visual feedback
        while True:
            # Handle different display modes
            if self.detection_mode == 0:  # Image-based detection
                with self.frame_lock:
                    if self.display_frame is not None:
                        # Add enhanced overlay and display
                        display_frame = self.add_status_overlay(self.display_frame.copy())
                        cv2.imshow(self.main_window, display_frame)
                    else:
                        # Show enhanced status image
                        cv2.imshow(self.main_window, status_img)
            else:  # Sound-based detection
                # Show enhanced audio status view
                audio_status_img = self.create_audio_status_image()
                cv2.imshow(self.main_window, audio_status_img)

            # Show enhanced log window
            log_img = self.create_log_image()
            cv2.imshow(self.log_window, log_img)

            # Process key presses with visual feedback
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == ord('s'):  # 's' to start/stop
                new_state = not self.is_running
                self.toggle_monitoring(new_state)
                cv2.setTrackbarPos("Monitor: OFF/ON", self.main_window, 1 if new_state else 0)
            elif key == ord('c'):  # 'c' to cleanup S3
                self.cleanup_s3()
                cv2.setTrackbarPos("Clean S3 Storage", self.main_window, 0)
            elif key == ord('m'):  # 'm' to switch mode
                new_mode = 1 if self.detection_mode == 0 else 0
                cv2.setTrackbarPos("Mode: Image/Sound", self.main_window, new_mode)

                # Update status image for new mode
                status_img = self.create_status_image()

            # Update status image periodically for animations/time
            if time.time() % 1 < 0.1:  # Update roughly every second
                status_img = self.create_status_image()

        # Cleanup
        self.stop_monitoring()
        cv2.destroyAllWindows()

    def create_status_image(self):
        """Create an improved status image when no camera feed is available"""
        status_img = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            alpha = i / 480.0
            color = (int(64 * alpha), int(45 * alpha), int(80 * alpha))
            status_img[i, :] = color

        # Mode display logic (fixing the '????' issue)
        mode_text = "MODE: IMAGE DETECTION" if self.detection_mode == 0 else "MODE: SOUND DETECTION"
        mode_icon = "🎥 " if self.detection_mode == 0 else "🔊 "
        
        # Add status header and mode text
        cv2.rectangle(status_img, (0, 0), (640, 70), (40, 40, 60), -1)
        cv2.putText(status_img, "Fall Detection System", (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(status_img, mode_icon + mode_text, (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 255), 2, cv2.LINE_AA)


        # Create status panel with rounded corners
        cv2.rectangle(status_img, (40, 100), (600, 250), (50, 50, 70), -1)
        cv2.rectangle(status_img, (40, 100), (600, 250), (80, 80, 100), 2)

        # Display status text dynamically based on current mode
        status_text = "MONITORING ACTIVE" if self.is_running else "MONITORING STOPPED"
        status_color = (0, 255, 0) if self.is_running else (0, 100, 255)
        cv2.putText(status_img, status_text, (150, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, status_color, 2, cv2.LINE_AA)

        # Mode: Add visual indicators for different detection modes
        mode_text = "MODE: IMAGE DETECTION" if self.detection_mode == 0 else "MODE: SOUND DETECTION"
        mode_icon = "🎥 " if self.detection_mode == 0 else "🔊 "
        cv2.putText(status_img, mode_icon + mode_text, (100, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (220, 220, 255), 2, cv2.LINE_AA)

        # Add fall count with a stylish visual emphasis
        fall_text = f"Falls Detected: {self.fall_count}"
        cv2.putText(status_img, fall_text, (150, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 200), 2, cv2.LINE_AA)

        # Add controls panel with a visually appealing border
        cv2.rectangle(status_img, (40, 280), (600, 430), (50, 50, 70), -1)
        cv2.rectangle(status_img, (40, 280), (600, 430), (80, 80, 100), 2)

        # Instructions with enhanced formatting
        instructions = [
            "[S] Start/Stop monitoring",
            "[M] Change detection mode",
            "[C] Cleanup S3 storage",
            "[Q/ESC] Exit application"
        ]

        y_pos = 340
        for instruction in instructions:
            cv2.putText(status_img, instruction, (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 220), 1, cv2.LINE_AA)
            y_pos += 30

        # Footer with enhanced branding
        cv2.rectangle(status_img, (0, 440), (640, 480), (40, 40, 60), -1)
        cv2.putText(status_img, "Raspberry Pi 5 Fall Detection v1.0", (180, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1, cv2.LINE_AA)

        return status_img

    def create_audio_status_image(self):
        """Create an improved image showing audio detection status"""
        status_img = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            alpha = i / 480.0
            color = (int(80 * alpha), int(45 * alpha), int(40 * alpha))
            status_img[i, :] = color

        # Mode display logic for audio status screen
        mode_text = "SOUND DETECTION MODE"  # We are always in sound mode here
        mode_icon = "🔊 "
        
        # Add header and mode text
        cv2.rectangle(status_img, (0, 0), (640, 70), (60, 40, 40), -1)
        cv2.putText(status_img, "Fall Detection System - Audio Mode", (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(status_img, mode_icon + mode_text, (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 255), 2, cv2.LINE_AA)


        # Create status panel
        cv2.rectangle(status_img, (40, 100), (600, 260), (50, 50, 70), -1)
        cv2.rectangle(status_img, (40, 100), (600, 260), (80, 80, 100), 2)

        # Add status text
        status_text = "SOUND MONITORING ACTIVE" if self.is_running else "SOUND MONITORING STOPPED"
        status_color = (0, 255, 0) if self.is_running else (0, 100, 255)
        cv2.putText(status_img, status_text, (100, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, status_color, 2, cv2.LINE_AA)

        # Draw circular indicator for status
        indicator_size = 15
        indicator_pos = (70, 140)
        cv2.circle(status_img, indicator_pos, indicator_size, status_color, -1)

        # Add sound detection icon
        cv2.putText(status_img, "🔊 SOUND DETECTION MODE", (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 255), 2, cv2.LINE_AA)

        # Add fall count
        cv2.putText(status_img, f"Falls Detected: {self.fall_count}", (150, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 200), 2, cv2.LINE_AA)

        # Add confidence level if monitoring is active
        if self.is_running:
            # Create audio level visualization
            cv2.rectangle(status_img, (120, 250), (520, 270), (40, 40, 60), -1)
            level_width = int(400 * min(self.audio_fall_confidence, 1.0))
            threshold_x = 120 + int(400 * 0.32)  # 0.32 is the threshold

            # Draw level bar
            level_color = (0, 100, 255) if self.audio_fall_confidence > 0.32 else (0, 255, 100)
            cv2.rectangle(status_img, (120, 250), (120 + level_width, 270), level_color, -1)

            # Draw threshold line
            cv2.line(status_img, (threshold_x, 245), (threshold_x, 275), (255, 255, 255), 2)

            # Add confidence text
            cv2.putText(status_img, f"Confidence: {self.audio_fall_confidence:.3f}", (150, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2, cv2.LINE_AA)

            # Add detection status text
            status_text = "FALL DETECTED" if self.audio_fall_confidence > 0.32 else "NO FALL"
            status_color = (0, 0, 255) if self.audio_fall_confidence > 0.32 else (0, 255, 0)
            cv2.putText(status_img, status_text, (350, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        # Add controls panel
        cv2.rectangle(status_img, (40, 330), (600, 430), (50, 50, 70), -1)
        cv2.rectangle(status_img, (40, 330), (600, 430), (80, 80, 100), 2)

        # Add control title
        cv2.putText(status_img, "Controls:", (60, 360), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (180, 180, 255), 1, cv2.LINE_AA)

        # Add instructions with improved layout
        instructions = [
            "[S] Start/Stop monitoring",
            "[M] Change detection mode",
            "[C] Cleanup S3 storage",
            "[Q/ESC] Exit application"
        ]

        # Reorganize the layout to avoid overlap
        y_pos = 390
        for i, instruction in enumerate(instructions):
            if i < 2:  # First column
                x_pos = 80
            else:  # Second column
                x_pos = 350

            cv2.putText(status_img, instruction, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 220), 1, cv2.LINE_AA)
            y_pos += 30

        # Add footer
        cv2.rectangle(status_img, (0, 440), (640, 480), (60, 40, 40), -1)
        cv2.putText(status_img, "Raspberry Pi 5 Fall Detection v1.0", (180, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 200), 1, cv2.LINE_AA)

        return status_img


    def add_status_overlay(self, frame):
        """Add improved status overlay to video frame"""
        # Add a semi-transparent overlay at the top for status info
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (40, 40, 60), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Add status text with better styling
        status_text = "MONITORING ACTIVE" if self.is_running else "MONITORING PAUSED"
        status_color = (0, 255, 0) if self.is_running else (0, 100, 255)

        # Add status indicator
        cv2.circle(frame, (20, 30), 10, status_color, -1)
        cv2.putText(frame, status_text, (40, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, status_color, 2, cv2.LINE_AA)

        # Add mode and fall count on same line with better spacing
        cv2.putText(frame, "🎥 IMAGE MODE", (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 255), 2, cv2.LINE_AA)

        # Add fall count with visual emphasis
        fall_text = f"Falls: {self.fall_count}"
        text_size = cv2.getTextSize(fall_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        fall_x = frame.shape[1] - text_size[0] - 20

        cv2.putText(frame, fall_text, (fall_x, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2, cv2.LINE_AA)

        # Add a subtle footer with timestamp
        time_str = datetime.now().strftime("%H:%M:%S")
        date_str = datetime.now().strftime("%Y-%m-%d")

        cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (40, 40, 60), -1)
        cv2.putText(frame, f"Time: {time_str}  Date: {date_str}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)

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
            self.is_running = True
            self.consecutive_fall_frames = 0

            if self.detection_mode == 0:  # Image-based detection
                # Initialize camera
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Unable to open camera.")

                self.log("Started image-based fall detection monitoring")

                # Start the video capture thread for continuous feed
                self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
                self.video_thread.start()

                # Start the image detection thread
                self.detection_thread = threading.Thread(target=self.image_detection_loop, daemon=True)
                self.detection_thread.start()

            else:  # Sound-based detection
                self.log("Started sound-based fall detection monitoring")

                # Start the sound detection thread
                self.detection_thread = threading.Thread(target=self.sound_detection_loop, daemon=True)
                self.detection_thread.start()

        except Exception as e:
            self.log(f"Error starting monitoring: {e}")
            self.stop_monitoring()

    def video_capture_loop(self):
        """Continuously capture video frames for display"""
        while self.is_running and self.detection_mode == 0:
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

    def image_detection_loop(self):
        """Image-based fall detection processing loop running in a separate thread"""
        # Frame counter for processing every Nth frame
        frame_counter = 0

        while self.is_running and self.detection_mode == 0:
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
                    self.log("ALERT: Fall detected from image!")

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
                self.log(f"Error in image detection loop: {e}")
                time.sleep(0.1)  # Small delay to prevent tight loop on errors

    def sound_detection_loop(self):
        """Sound-based fall detection loop running in a separate thread"""
        while self.is_running and self.detection_mode == 1:
            try:
                # Predict fall from audio
                is_fall = self.predict_fall_from_audio()

                if is_fall:
                    self.fall_count += 1
                    self.log("ALERT: Fall detected from sound!")

                    # Create a visual representation of the sound detection
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

                    # Create a simple visual for the sound detection
                    sound_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(sound_img, "FALL DETECTED BY SOUND", (80, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(sound_img, f"Confidence: {self.audio_fall_confidence:.4f}", (150, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(sound_img, timestamp, (180, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 255), 2, cv2.LINE_AA)
                    cv2.circle(sound_img, (320, 350), 50, (0, 0, 255), -1)

                    # Save the visual representation as an image
                    save_path = os.path.join(self.capture_dir, f"fall_sound_{timestamp}.jpg")
                    cv2.imwrite(save_path, sound_img)

                    self.log(f"Sound fall detection image saved: {save_path}")


                    # Now, record the audio and save it to a file
                    audio = self.record_audio_in_memory()  # Record the fall sound
                    wav_file_path = os.path.join(self.capture_dir, f"fall_sound_{timestamp}.wav")
                    self.save_audio_to_file(audio, wav_file_path)  # Save the audio to WAV file

                    # Convert WAV to MP3
                    mp3_file_path = self.convert_wav_to_mp3(wav_file_path)

                    # Upload the MP3 audio to S3
                    audio_s3_key = f"fall_sound_{timestamp}.mp3"
                    upload_thread_audio = threading.Thread(
                        target=self.upload_to_s3,
                        args=(mp3_file_path, audio_s3_key),
                        daemon=True
                    )
                    upload_thread_audio.start()

                    # Pause between recordings
                    time.sleep(1)  # Short delay before taking the next audio sample

            except Exception as e:
                self.log(f"Error in sound detection loop: {e}")
                time.sleep(0.5)  # Longer delay on error

    def save_audio_to_file(self, audio, file_path):
        """Save the recorded audio to a WAV file"""
        write(file_path, self.sample_rate, audio)
        self.log(f"Audio saved to {file_path}")

    def convert_wav_to_mp3(self, wav_file_path):
        """Convert the WAV file to MP3 format"""
        mp3_file_path = wav_file_path.replace('.wav', '.mp3')
        
        # Load the WAV file using pydub
        audio = AudioSegment.from_wav(wav_file_path)
        
        # Export the audio to MP3 format
        audio.export(mp3_file_path, format="mp3")
        self.log(f"Audio converted to MP3: {mp3_file_path}")
        
        # Optionally, remove the WAV file after conversion
        os.remove(wav_file_path)
        
        return mp3_file_path


    def stop_monitoring(self):
        """Stop the fall detection monitoring"""
        self.is_running = False

        # Wait for threads to finish
        thread_names = ['video_thread', 'detection_thread']
        for thread_name in thread_names:
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

        # Log the mode that was stopped
        mode_name = "image-based" if self.detection_mode == 0 else "sound-based"
        self.log(f"Stopped {mode_name} fall detection monitoring")


if __name__ == "__main__":
    app = FallDetectionApp()
    app.run()
