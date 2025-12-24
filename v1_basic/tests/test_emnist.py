import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import sys

from pathlib import Path

# Replace with the actual path to your model
APP_DIR = Path(__file__).resolve().parent.parent  # Adjust as needed
EMNIST_MODEL_PATH = APP_DIR / "model" / "emnist_byclass_cnn_model.h5"

# EMNIST labels (same as your API)
EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
]

def preprocess_image(image_path):
    """Load image, convert to grayscale, resize to 28x28, normalize, add batch & channel dims."""
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0  # normalize to [0,1]
    img_array = 1.0 - img_array  # invert colors if needed (white bg, black text)
    img_array = img_array.reshape(1, 28, 28, 1)  # batch and channel dims
    return img_array

def main(image_path):
    model = load_model(EMNIST_MODEL_PATH)
    x = preprocess_image(image_path)

    probs = model.predict(x)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    predicted_char = EMNIST_LABELS[pred_idx]

    print(f"Predicted character: {predicted_char} (class={pred_idx}, confidence={confidence:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_emnist.py path_to_image.png")
    else:
        main(sys.argv[1])
