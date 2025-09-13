import os
import shutil
import random

# -----------------------
# Paths
# -----------------------
dataset_dir = "data/nhcd"   # raw dataset after lifting
output_dir  = "data/processed_dataset"

train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

# -----------------------
# Create output folders
# -----------------------
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# -----------------------
# Loop over first-level folders (consonants, vowels, numerals)
# -----------------------
for first_level in os.listdir(dataset_dir):
    first_level_path = os.path.join(dataset_dir, first_level)
    if not os.path.isdir(first_level_path):
        continue

    # Loop over character folders
    for category in os.listdir(first_level_path):
        class_path = os.path.join(first_level_path, category)
        if not os.path.isdir(class_path):
            continue

        # Collect all images (supported extensions)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        if len(images) == 0:
            print(f"⚠️ No images found in {class_path}")
            continue

        random.shuffle(images)
        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val   = int(val_ratio * n_total)

        train_files = images[:n_train]
        val_files   = images[n_train:n_train+n_val]
        test_files  = images[n_train+n_val:]

        # Copy images to train/val/test preserving nested structure
        for split_name, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_class_dir = os.path.join(output_dir, split_name, first_level, category)
            os.makedirs(split_class_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(class_path, f), os.path.join(split_class_dir, f))

        print(f"✅ Processed category: {first_level}/{category} | Total images: {n_total}")

print("🎉 Dataset split done! Nested structure preserved.")
