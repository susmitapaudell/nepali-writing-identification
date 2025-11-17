import numpy as np
import cv2
import tensorflow as tf

# -------------------
# Paths
# -------------------
MODEL_PATH = "models/nepali_cnn.h5"
LABELS_PATH = "models/class_labels.npy"

# -------------------
# Load Model + Labels
# -------------------
model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True).item()

# -------------------
# Preprocess Image
# -------------------
def preprocess_image(img_path, img_size=28):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # channel dimension
    img = np.expand_dims(img, axis=0)   # batch dimension
    return img

# -------------------
# Predict
# -------------------
def predict_character(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return labels[class_idx]

# -------------------
# Test
# -------------------
if __name__ == "__main__":
    #test_img = "images/char.png"  # your image path
    test_img = "/Users/susmitapaudel/projects/nepali-writing-identification/images/s5.png"  # your image path
    result = predict_character(test_img)
    print(f"Predicted character: {result}")
