# v2_line_ocr

Single-line handwritten text recognition (CRNN) — small, fast, offline OCR for single-line inputs.

Summary
- Lightweight CRNN model for single-line handwriting.
- Fast inference via FastAPI backend.
- Simple React + Vite frontend for uploading line images and viewing recognized text.

Repository layout
- `backend/` — FastAPI server; saves uploads to `uploads/` and returns recognized text via `/ocr`.
  - `main.py` — FastAPI app
  - `ocr.py` — loader for ML module (resolves model/charset paths)
- `ml/` — model and inference code
  - `model.py` — CRNN definition
  - `infer.py` — inference pipeline and preprocessing
  - `crnn_line_v2.pth` — model weights (binary)
  - `charset.json` — indexed character set used by the model
- `frontend/` — React + Vite UI
  - `src/` — React components and styles
  - `package.json` — frontend dependencies and scripts

Prerequisites
- Python 3.8+
- pip
- Node 16+/npm (for frontend)
- (Optional) GPU + CUDA for faster inference
- Git and a GitHub account (optional `gh` CLI if you want to create remote repos from CLI)

Backend — setup & run (PowerShell)
1. Create and activate venv, then install:
   ```powershell
   cd v2_line_ocr\backend
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r ..\requirements.txt