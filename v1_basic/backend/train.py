# train.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ---------------------------
# 1. Load MNIST dataset
# ---------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape → (28,28,1)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# ---------------------------
# 2. Data Augmentation
# ---------------------------
# This helps model generalize better for handwritten styles
datagen = ImageDataGenerator(
    rotation_range=10,        # rotate digits
    width_shift_range=0.1,    # shift horizontally
    height_shift_range=0.1,   # shift vertically
    zoom_range=0.1            # zoom in/out
)
datagen.fit(x_train)

# ---------------------------
# 3. Build a stronger CNN
# ---------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation="relu"),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

# ---------------------------
# 4. Compile
# ---------------------------
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ---------------------------
# 5. Train
# ---------------------------
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_test, y_test),
    epochs=12,
    verbose=1
)

# ---------------------------
# 6. Save model
# ---------------------------
os.makedirs("model", exist_ok=True)
model.save("model/mnist_cnn.h5")
print("✅ Model saved at model/mnist_cnn.h5")

# ---------------------------
# 7. Evaluate final accuracy
# ---------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"🎯 Final Test Accuracy: {test_acc*100:.2f}%")
