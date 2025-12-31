import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ML_DIR = os.path.join(PROJECT_ROOT, "ml")

sys.path.insert(0, ML_DIR)

from infer import HandwritingRecognizer  # type: ignore

# Use absolute paths
model_path = os.path.join(ML_DIR, "crnn_line_v2.pth")
charset_path = os.path.join(ML_DIR, "charset.json")

ocr_engine = HandwritingRecognizer(model_path, charset_path)


def run_ocr(image_path: str) -> str:
    return ocr_engine.recognize(image_path)