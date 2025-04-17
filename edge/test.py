import sounddevice as sd
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
import cv2

# Load the trained model
model = tf.keras.models.load_model("best_model3.keras")

# Settings
duration = 3  # seconds
sample_rate = 16000
resize_shape = (64, 64)

def record_audio_in_memory():
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return audio.squeeze()  # Remove extra dimension

def process_audio_array(audio, sr):
    if audio.ndim > 1:
        audio = audio[:, 0]
    freqs, times, spec = spectrogram(audio, fs=sr)
    log_spec = 10 * np.log10(spec + 1e-10)
    resized = cv2.resize(log_spec, resize_shape)
    normalized = (resized - resized.min()) / (resized.max() - resized.min())
    return normalized[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

def predict_fall_from_audio():
    # Record audio
    audio = record_audio_in_memory()

    # Get the spectrogram
    spec_input = process_audio_array(audio, sample_rate)  # shape: (1, 64, 64, 1)

    # Predict using the spectrogram input only
    prediction = model.predict(spec_input, verbose=0)[0][0]

    print(f"🔎 Prediction Confidence: {prediction:.4f}")
    if prediction > 0.5:
        print("⚠️ FALL detected!")
        return True
    else:
        print("✅ No fall detected.")
        return False

try:
    while True:
        is_fall = predict_fall_from_audio()
        if is_fall:
            print("⚠️ Detected: FALL")
        else:
            print("✅ No fall detected this round.")

except KeyboardInterrupt:
    print("\n🛑 Monitoring stopped by user.")
