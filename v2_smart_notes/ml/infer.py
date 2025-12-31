import torch
import cv2
import json
import numpy as np
from model import CRNN


class HandwritingRecognizer:
    def __init__(self, model_path, charset_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load charset
        with open(charset_path, "r") as f:
            self.chars = json.load(f)

        # Load model
        self.model = CRNN(len(self.chars))
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image_path: str) -> str:
        # --- LOAD IMAGE ---
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""

        # --- IMPROVED PREPROCESSING ---
        # Invert colors (white text on black background)
        img = cv2.bitwise_not(img)

        # Adaptive thresholding for better contrast
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Light denoising
        img = cv2.medianBlur(img, 3)

        # Normalize contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Resize maintaining aspect ratio
        h, w = img.shape
        new_w = int(w * (32 / h))
        # Ensure minimum width for better recognition
        new_w = max(new_w, 64)
        img = cv2.resize(img, (new_w, 32), interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Tensor: (1, 1, 32, W)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = img.to(self.device)

        # --- INFERENCE ---
        with torch.no_grad():
            logits = self.model(img)
            # Get confidence scores and predictions
            probs = logits.softmax(2)
            preds = probs.argmax(2)[0]
            confidences = probs[0].max(1)[0]

        # --- IMPROVED CTC DECODE WITH CONFIDENCE FILTERING ---
        result = []
        prev = -1
        confidence_threshold = 0.3  # Filter low-confidence predictions

        for idx, p in enumerate(preds):
            p_idx = p.item()
            conf = confidences[idx].item()

            # Only add if:
            # 1. Not the blank token (0)
            # 2. Different from previous prediction (CTC collapse)
            # 3. Above confidence threshold
            if p_idx != 0 and p_idx != prev and conf > confidence_threshold:
                result.append(self.chars[p_idx])

            prev = p_idx

        return "".join(result)