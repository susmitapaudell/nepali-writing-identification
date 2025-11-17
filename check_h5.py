from tensorflow.keras.models import load_model

model = load_model("models/nepali_cnn.h5")
model.summary()

import numpy as np
data = np.load("models/class_labels.npy", allow_pickle=True)
print(data)