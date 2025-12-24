# scripts/make_letters.py
import numpy as np
import cv2
from pathlib import Path
import random
import string

# --- Setup paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)
out_path = SAMPLES_DIR / "multi_letters.png"

# --- Generate a random string of letters ---
# EMNIST letters: A-Z and a-z (uppercase and lowercase)
letters = string.ascii_letters  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
num_letters = 6  # how many letters to generate

random_letters = ''.join(random.choice(letters) for _ in range(num_letters))

print(f"Generating image with letters: {random_letters}")

# --- Create white canvas ---
canvas_width = 40 * num_letters + 20
canvas = np.ones((60, canvas_width), dtype=np.uint8) * 255  # white background

# --- Draw letters ---
x = 10
for ch in random_letters:
    cv2.putText(
        canvas, ch, (x, 45),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=(0,),  # black
        thickness=4,
        lineType=cv2.LINE_AA
    )
    x += 40

# --- Save image ---
cv2.imwrite(str(out_path), canvas)
print(f"✅ Saved multi-letter image at: {out_path}")
