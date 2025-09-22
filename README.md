Handwritten Text Recognition

A project for recognizing handwritten digits, letters, and full-line text using deep learning and OCR. This project leverages MNIST and EMNIST datasets for training models and uses PaddleOCR for line-level text recognition.

---

Features

- Digit recognition (0–9) using a convolutional neural network trained on MNIST
- Letter recognition (A–Z, a–z) using EMNIST ByClass dataset
- Line-level text recognition using PaddleOCR
- Modular and testable code structure
- Python backend scripts for training, preprocessing, and inference

---

Project Structure

Handwritten-Text-Recognition/
├── data/ # Raw datasets (MNIST, EMNIST)
├── models/ # Trained models and checkpoints
├── notebooks/ # Jupyter notebooks for experimentation
├── src/
│ ├── preprocessing/ # Image preprocessing scripts
│ ├── training/ # Model training code
│ ├── inference/ # Prediction scripts
├── samples/ # Example images for testing
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Files/folders to ignore in Git


---

Installation

1. Clone the repository:


git clone https://github.com/ZalakRajvanshi/Handwritten-Text-Recognition.git
cd Handwritten-Text-Recognition
Create a virtual environment:


python -m venv venv
Activate the virtual environment:

On Windows:


.\venv\Scripts\activate
On macOS/Linux:


source venv/bin/activate
Install Python dependencies:


pip install -r requirements.txt
Usage

Training a Model
python src/training/train.py --dataset emnist

Running Inference
python src/inference/predict.py --image samples/sample.png

Testing
To run unit tests:


pytest tests/
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
MNIST and EMNIST datasets

PaddleOCR for line-level text recognition

Open-source Python libraries used for model development




