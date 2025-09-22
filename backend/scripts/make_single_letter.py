import numpy as np
import cv2
from pathlib import Path

# --- Setup paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT_DIR / "samples" / "single_letters"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# List of characters to generate single letter images for
chars = list("Z9bY7a")  # example chars; modify as needed

for ch in chars:
    # Create white 28x28 canvas (EMNIST input size)
    canvas = np.ones((28, 28), dtype=np.uint8) * 255  # white background

    # Get text size to center it
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    (w, h), _ = cv2.getTextSize(ch, font, font_scale, thickness)
    x = (28 - w) // 2
    y = (28 + h) // 2

    # Put the character in black
    cv2.putText(canvas, ch, (x, y), font, font_scale, 0, thickness, cv2.LINE_AA)

    # Save image as PNG (grayscale)
    out_path = SAMPLES_DIR / f"{ch}.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved '{ch}' to {out_path}")
