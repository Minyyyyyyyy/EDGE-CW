import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate

# Parse filename and extract metadata (only label for fall or non-fall)
def parse_filename(filename):
    parts = filename.replace(".wav", "").split("-")
    label = 1 if parts[-1] == "01" else 0  # fall = 1, non-fall = 0
    group_key = "-".join(parts[:-1])  # for GroupKFold
    return label, group_key

# Convert audio to spectrogram
def compute_spectrogram(file_path):
    sr, audio = wavfile.read(file_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    freqs, times, spec = spectrogram(audio, fs=sr)
    log_spec = 10 * np.log10(spec + 1e-10)
    resized = cv2.resize(log_spec, (64, 64))
    normalized = (resized - resized.min()) / (resized.max() - resized.min())
    return normalized

# Prepare dataset
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

# Define multi-input model
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

# Main
if __name__ == "__main__":
    dataset_dir = r"C:\Users\seyon\Downloads\archive (5)"
    X_spec, y, groups = prepare_dataset(dataset_dir)

    # GroupKFold cross-validation
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X_spec, y, groups):
        X_train_spec, X_val_spec = X_spec[train_idx], X_spec[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        break  # use the first fold only

    model = build_model()
    checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model2.keras", monitor='val_accuracy', save_best_only=True, verbose=1)

    model.fit(X_train_spec, y_train,
              validation_data=(X_val_spec, y_val),
              epochs=20, batch_size=32, callbacks=[checkpoint])

    loss, acc = model.evaluate(X_val_spec, y_val)
    print(f"\n✅ True Test Accuracy (Group-split): {acc:.4f}")
