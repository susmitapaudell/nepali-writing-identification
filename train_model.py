import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------
# Settings
# -------------------
DATA_DIR = "data/processed_dataset/train"   # Point directly to train folder
IMG_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 60
MODEL_PATH = "models/nepali_cnn.h5"

# -------------------
# Mapping nested folders to actual letters
# CHANGE THIS according to your dataset
nested_mapping = {
    "vowels": {
        "1": "a",      # अ
        "2": "aa",     # आ
        "3": "i",      # इ
        "4": "ii",     # ई
        "5": "u",      # उ
        "6": "uu",     # ऊ
        "7": "e",      # ए
        "8": "ai",     # ऐ
        "9": "o",     # ओ
        "10": "au",    # औ
        "11": "am",    # अं
        "12": "ah",    # अः
    },
    "consonants": {
        "1": "ka", "2": "kha", "3": "ga", "4": "gha", "5": "nga",
        "6": "cha", "7": "chha", "8": "ja", "9": "jha", "10": "nya",
        "11": "tta", "12": "ttha", "13": "dda", "14": "ddha", "15": "nna",
        "16": "ta", "17": "tha", "18": "da", "19": "dha", "20": "na",
        "21": "pa", "22": "pha", "23": "ba", "24": "bha", "25": "ma",
        "26": "ya", "27": "ra", "28": "la", "29": "wa",
        "30": "sha", "31": "ssha", "32": "sa", "33": "ha",
        "34": "ksha", "35": "tra", "36": "gya"
    },
    "numerals": {
        "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
        "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"
    }
}


# -------------------
# Create a flat dataset structure temporarily for Keras
flat_dir = "data/flat_train"
os.makedirs(flat_dir, exist_ok=True)

for top_folder, subfolders in nested_mapping.items(): #vowels, consonants chai top_folder bhayo, 01,02 haru chai subfolders bho
    for subfolder, letter in subfolders.items():
        src = os.path.join(DATA_DIR, top_folder, subfolder)
        dst = os.path.join(flat_dir, letter)
        if not os.path.isdir(src):
            print(f"[SKIP] Directory does not exist: {src}")
            continue
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))


# -------------------
# Data Generator
# -------------------
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_data = datagen.flow_from_directory(
    flat_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    flat_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# -------------------
# Model
# -------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------
# Train
# -------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------
# Save Model + Labels
# -------------------
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

# Save class labels
labels = {v: k for k, v in train_data.class_indices.items()}
np.save("models/class_labels.npy", labels)

print("✅ Model and labels saved!")

import matplotlib.pyplot as plt

# Plot training & validation loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
plt.title('Model Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training & validation accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.title('Model Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
