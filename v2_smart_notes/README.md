# v2 – Single-Line Handwritten OCR (CRNN)

This project is a **working, end-to-end handwritten OCR system**
designed specifically for **single-line handwritten text recognition**.

It demonstrates how a trained ML model can be integrated into a real system
with preprocessing, inference, API, and frontend.

---

## 🚀 Features

- CRNN-based handwritten OCR
- Trained on the **IAM Line** dataset
- Proper preprocessing alignment with training
- Greedy CTC decoding
- FastAPI backend for inference
- React frontend for live demo
- Fully offline (no external APIs)

---

## 🧠 Model Overview

- **Architecture**: CRNN (CNN + BiLSTM + CTC)
- **Input**: Single-line handwritten image
- **Output**: Recognized text string
- **Loss**: CTC Loss
- **Decoding**: Greedy CTC decoding

This model focuses on **character-level recognition** and does not use
any language model or spell correction.


## 📌 Project Scope

This project focuses on single-line handwritten text recognition, enabling a clean, efficient, and reliable OCR pipeline.
The narrow scope ensures strong character-level accuracy and provides a solid foundation for future extensions.



## 🖥️ System Architecture

Image Upload
↓
Preprocessing (resize, normalize, invert)
↓
CRNN Model
↓
CTC Decoding
↓
FastAPI Endpoint
↓
React Frontend




## 🔧 Tech Stack

- **Python**
- **PyTorch**
- **OpenCV**
- **FastAPI**
- **React (Vite)**



## ▶️ How to Run
---

### Backend

cd backend

uvicorn main:app --reload

### Frontend

cd frontend

npm install

npm run dev


Open browser:
http://localhost:5173


🎓 What This Project Demonstrates

Training and deploying ML models

Debugging model–architecture mismatches

Importance of preprocessing consistency

Honest problem scoping

End-to-end AI system building


🔜 Future Work


Add beam search decoding

Integrate language model

Extend to full-page handwritten notes (v3)


👤 Author
Zalak Rajvanshi
AI / ML Developer
Focus: Applied Machine Learning & OCR Systems






