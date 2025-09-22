from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import tensorflow as tf
import tempfile
import os
import traceback
import logging
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from backend.utils import prepare_single_from_bytes, segment_digits_from_bytes, visualize_boxes

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent.parent
MNIST_MODEL_PATH = APP_DIR / "model" / "mnist_cnn.h5"
EMNIST_MODEL_PATH = APP_DIR / "model" / "emnist_byclass_cnn_model.h5"

mnist_model = None
emnist_model = None
ocr_engine = None

# EMNIST BYCLASS: 62 classes (0–9, A–Z, a–z)
EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: loads models and OCR engine on startup, cleans up on shutdown.
    """
    global mnist_model, emnist_model, ocr_engine

    if MNIST_MODEL_PATH.exists():
        mnist_model = tf.keras.models.load_model(str(MNIST_MODEL_PATH))
        mnist_model(np.zeros((1, 28, 28, 1), dtype="float32"), training=False)
        logging.info("MNIST model loaded ✅")

    if EMNIST_MODEL_PATH.exists():
        emnist_model = tf.keras.models.load_model(str(EMNIST_MODEL_PATH))
        emnist_model(np.zeros((1, 28, 28, 1), dtype="float32"), training=False)
        logging.info("EMNIST model loaded ✅")

    if PaddleOCR:
        try:
            ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
            logging.info("PaddleOCR engine initialized ✅")
        except Exception as e:
            logging.error(f"❌ PaddleOCR initialization failed: {e}")
            ocr_engine = None

    yield  # App runs here


# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(
    title="Handwritten Recognition API",
    version="2.0.0",
    lifespan=lifespan
)

# ------------------------
# CORS Middleware
# ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------
# Routes
# ------------------------
@app.get("/")
async def root():
    return {"message": "✅ Handwritten Recognition API is running! Use /ocr/ to upload images."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-digit")
async def predict_digit(file: UploadFile = File(...)):
    if mnist_model is None:
        raise HTTPException(status_code=500, detail="MNIST model not loaded.")

    x = prepare_single_from_bytes(await file.read())
    probs = mnist_model(x, training=False).numpy()[0]
    return {
        "predicted_digit": int(np.argmax(probs)),
        "confidence": float(np.max(probs))
    }


@app.post("/predict-letter")
async def predict_letter(file: UploadFile = File(...)):
    if emnist_model is None:
        raise HTTPException(status_code=500, detail="EMNIST model not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")
    x = prepare_single_from_bytes(await file.read())

    probs = emnist_model(x, training=False).numpy()[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    predicted_char = EMNIST_LABELS[pred_idx] if pred_idx < len(EMNIST_LABELS) else "?"

    return {
        "predicted_label": pred_idx,
        "predicted_character": predicted_char,
        "confidence": confidence
    }


@app.post("/predict-ocr")
async def predict_ocr(file: UploadFile = File(...)):
    if ocr_engine is None:
        raise HTTPException(status_code=500, detail="PaddleOCR not available.")

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = ocr_engine.ocr(str(tmp_path))

        try:
            os.remove(tmp_path)
        except Exception as cleanup_error:
            logging.warning(f"Cleanup failed: {cleanup_error}")

        texts = [{"text": line[1][0], "confidence": float(line[1][1])} for line in result[0]]
        return {"ocr_result": texts}

    except Exception as e:
        logging.error("🔥 OCR Error:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"OCR failed: {str(e)}"},
        )


@app.post("/debug-segmentation")
async def debug_segmentation(file: UploadFile = File(...)):
    """
    Debug endpoint: returns detected bounding boxes and saves a debug image with boxes drawn.
    """
    try:
        contents = await file.read()
        batch, boxes = segment_digits_from_bytes(contents)

        if not boxes:
            return {"boxes": [], "message": "No digits/letters detected"}

        debug_path = APP_DIR / "debug_boxes.png"
        visualize_boxes(contents, boxes, str(debug_path))

        return {
            "boxes": boxes,
            "count": len(boxes),
            "debug_image": str(debug_path)
        }
    except Exception as e:
        logging.error("Segmentation debug failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
