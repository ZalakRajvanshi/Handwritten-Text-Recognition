import sys
import os
import numpy as np
from pathlib import Path
import logging
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.utils import (
    prepare_single_from_bytes,
    segment_digits_from_bytes,
    prepare_for_ocr,
    visualize_boxes,
    prepare_single_from_path,
)

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent.parent
MNIST_MODEL = ROOT / "model" / "mnist_cnn.h5"
EMNIST_MODEL = ROOT / "model" / "emnist_byclass_cnn_model.h5"

# Load models once instead of every prediction
_mnist_model = load_model(str(MNIST_MODEL)) if MNIST_MODEL.exists() else None
_emnist_model = load_model(str(EMNIST_MODEL)) if EMNIST_MODEL.exists() else None

EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


def predict_mnist(image_path: str, debug_out: str = None):
    if _mnist_model is None:
        logging.error("MNIST model not found.")
        return

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    with open(image_path, "rb") as f:
        batch, boxes = segment_digits_from_bytes(f.read())


    if debug_out and boxes:
        visualize_boxes(image_bytes, boxes, debug_out)
        logging.info(f"Saved debug visualization -> {debug_out}")

    if batch.shape[0] == 0:
        x = prepare_single_from_path(image_path)
        probs = _mnist_model(x, training=False).numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        print(f"[MNIST] Single-digit -> {pred} (conf={conf:.4f})")
        return

    preds = np.argmax(_mnist_model(batch, training=False).numpy(), axis=1).tolist()
    print(f"[MNIST] Sequence -> {''.join(map(str, preds))} (detected {len(boxes)} digits)")


def predict_emnist(image_path: str):
    if _emnist_model is None:
        logging.error("EMNIST model not found.")
        return

    with open(image_path, "rb") as f:
        x = prepare_single_from_bytes(f.read())
    probs = _emnist_model(x, training=False).numpy()[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = EMNIST_LABELS[pred] if pred < len(EMNIST_LABELS) else "?"

    print(f"[EMNIST BYCLASS] Predicted -> {label} (class={pred}, conf={conf:.4f})")


def predict_ocr(image_path: str):
    if PaddleOCR is None:
        logging.error("PaddleOCR not installed.")
        return

    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    result = ocr.ocr(str(image_path))

    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        print(f"[OCR] {text} (conf={conf:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict using MNIST, EMNIST, or PaddleOCR")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("--mode", type=str, choices=["mnist", "emnist", "ocr"], default="mnist")
    parser.add_argument("--debug", type=str, help="Optional path to save debug visualization with boxes")
    args = parser.parse_args()

    if args.mode == "mnist":
        predict_mnist(args.image, debug_out=args.debug)
    elif args.mode == "emnist":
        predict_emnist(args.image)
    elif args.mode == "ocr":
        predict_ocr(args.image)
