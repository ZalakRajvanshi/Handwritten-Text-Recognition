import numpy as np
import cv2
from typing import Tuple, List

TARGET_SIZE = (28, 28)

# ---------------------------
# Shared helpers
# ---------------------------
def _to_gray(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Use PNG/JPG/BMP.")
    return img

def _to_rgb(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Use PNG/JPG/BMP.")
    return img

def _invert_if_needed(gray: np.ndarray) -> np.ndarray:
    # Use corners to estimate background instead of mean
    h, w = gray.shape
    corners = [gray[0,0], gray[0,w-1], gray[h-1,0], gray[h-1,w-1]]
    bg = np.median(corners)
    return 255 - gray if bg < 127 else gray

def _pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    size = max(h, w)
    out = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    out[y0:y0+h, x0:x0+w] = img
    return out

def _deskew(img: np.ndarray, skew_threshold: float = 0.01) -> np.ndarray:
    try:
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.moments(bw)
        if abs(m["mu02"]) < 1e-2:
            return img
        skew = m["mu11"] / (m["mu02"] + 1e-8)
        if abs(skew) < skew_threshold:
            return img
        M = np.float32([[1, skew, -0.5 * img.shape[1] * skew], [0, 1, 0]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    except Exception as e:
        print(f"[WARN] Deskewing failed: {e}")
        return img


def _normalize_contrast(img: np.ndarray) -> np.ndarray:
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)
    except Exception:
        return img


# ---------------------------
# MNIST + EMNIST preprocessing
# ---------------------------
def prepare_single_from_bytes(image_bytes: bytes, target_size=TARGET_SIZE) -> np.ndarray:
    gray = _to_gray(image_bytes)
    gray = _invert_if_needed(gray)
    gray = _normalize_contrast(gray)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(th)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        crop = gray[y:y+h, x:x+w]
    else:
        crop = np.zeros(target_size, dtype=np.uint8)

    crop = _pad_to_square(crop)
    crop = _deskew(crop)
    crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    crop = _normalize_contrast(crop)

    x = crop.astype("float32") / 255.0
    x = x.reshape(1, target_size[0], target_size[1], 1)
    return x

# ---------------------------
# OCR preprocessing (PaddleOCR needs RGB)
# ---------------------------
def prepare_for_ocr(image_bytes: bytes) -> np.ndarray:
    img = _to_rgb(image_bytes)
    # Optional: denoise & normalize for better OCR
    img = cv2.medianBlur(img, 3)
    return img

# ---------------------------
# Multi-digit segmentation (MNIST style)
# ---------------------------
def segment_digits_from_bytes(image_bytes: bytes) -> Tuple[np.ndarray, list]:
    gray = _to_gray(image_bytes)
    gray = _invert_if_needed(gray)
    gray = _normalize_contrast(gray)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, patches = [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 3 or h < 3:  # less strict than w*h < 64
            continue
        boxes.append((x,y,w,h))
    if not boxes:
        return np.zeros((0,28,28,1), dtype="float32"), []
    boxes.sort(key=lambda b: b[0])

    for x,y,w,h in boxes:
        roi = gray[y:y+h, x:x+w]
        roi = _pad_to_square(roi)
        roi = _deskew(roi)
        roi = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        roi = _normalize_contrast(roi)
        arr = roi.astype("float32") / 255.0
        patches.append(arr.reshape(28,28,1))

    batch = np.stack(patches, axis=0)
    return batch, boxes

# ---------------------------
# Debug visualization helper
# ---------------------------
def visualize_boxes(image_bytes: bytes, boxes: List[Tuple[int,int,int,int]], out_path: str) -> None:
    """
    Save an image with bounding boxes drawn (for debugging segmentation).
    """
    img = _to_rgb(image_bytes)
    for (x,y,w,h) in boxes:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imwrite(out_path, img)
