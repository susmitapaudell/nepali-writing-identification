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
EPOCHS = 10
MODEL_PATH = "models/nepali_cnn.h5"

# -------------------
# Mapping nested folders to actual letters
# CHANGE THIS according to your dataset
nested_mapping = {
    'consonants': {'1': 'ka', '2': 'kha', '3': 'ga'},
    'vowels': {'1': 'a', '2': 'aa'}
}

# -------------------
# Create a flat dataset structure temporarily for Keras
flat_dir = "data/flat_train"
os.makedirs(flat_dir, exist_ok=True)

for top_folder, subfolders in nested_mapping.items():
    for subfolder, letter in subfolders.items():
        src = os.path.join(DATA_DIR, top_folder, subfolder)
        dst = os.path.join(flat_dir, letter)
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
