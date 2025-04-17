import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# === Parse filename and extract metadata ===
def parse_filename(filename):
    parts = filename.replace(".wav", "").split("-")
    label = 1 if parts[-1] == "01" else 0  # 01 = fall, 02 = non-fall
    group_key = "-".join(parts[:-1])      # use all but label as group key
    return label, group_key

# === Convert audio file to spectrogram ===
def compute_spectrogram(file_path):
    sr, audio = wavfile.read(file_path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # Use only one channel if stereo
    freqs, times, spec = spectrogram(audio, fs=sr)
    log_spec = 10 * np.log10(spec + 1e-10)  # log scale
    resized = cv2.resize(log_spec, (64, 64))
    normalized = (resized - resized.min()) / (resized.max() - resized.min())
    return normalized

# === Load and preprocess dataset ===
def prepare_dataset(directory):
    X_spec, y, groups = [], [], []

    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            try:
                label, group_key = parse_filename(filename)
                spec = compute_spectrogram(filepath)
                X_spec.append(spec)
                y.append(label)
                groups.append(group_key)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    X_spec = np.array(X_spec)[..., np.newaxis]
    y = np.array(y)
    return X_spec, y, np.array(groups)

# === Define CNN model ===
def build_model():
    input_spec = Input(shape=(64, 64, 1))
    x = Conv2D(16, (3, 3), activation='relu')(input_spec)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_spec, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Main block ===
if __name__ == "__main__":
    dataset_dir = r"C:\Users\seyon\Downloads\archive (5)"  # Adjust path
    X_spec, y, groups = prepare_dataset(dataset_dir)

    # Use GroupKFold for speaker/environment-wise split
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X_spec, y, groups):
        X_train_spec, X_val_spec = X_spec[train_idx], X_spec[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        break  # Use the first fold only

    # Build and train model
    model = build_model()
    checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model4.keras", monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(X_train_spec, y_train,
              validation_data=(X_val_spec, y_val),
              epochs=20, batch_size=32, callbacks=[checkpoint])

    # Evaluate model
    loss, acc = model.evaluate(X_val_spec, y_val)
    print(f"\n✅ True Test Accuracy (Group-split): {acc:.4f}")

    # Detailed evaluation
    y_pred_prob = model.predict(X_val_spec)
    y_pred = (y_pred_prob > 0.47).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["Non-Fall", "Fall"]))

    cm = confusion_matrix(y_val, y_pred)
    print("🔍 Confusion Matrix:")
    print(cm)

    # Optional: Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fall", "Fall"], yticklabels=["Non-Fall", "Fall"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
