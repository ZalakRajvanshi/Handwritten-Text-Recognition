# scripts/make_multi.py
import numpy as np
import cv2
from pathlib import Path

# --- Setup paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)
out_path = SAMPLES_DIR / "multi.png"

# --- Create a fake multi-digit image ---
# white background
canvas = np.ones((60, 200), dtype=np.uint8) * 255  

# choose some digits to "write"
digits = ["7", "1", "9", "3"]

x = 10
for d in digits:
    cv2.putText(canvas, d, (x, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0,), 4, cv2.LINE_AA)
    x += 40

# --- Save result ---
cv2.imwrite(str(out_path), canvas)
print(f"Wrote {out_path}")