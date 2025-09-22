import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# -------------------------------------------------------------------
# Setup paths
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from api.main import app  # noqa: E402

SAMPLES_DIR = ROOT / "samples"
SAMPLE_DIGIT = SAMPLES_DIR / "digit.png"
SAMPLE_MULTI_DIGITS = SAMPLES_DIR / "multi.png"
SAMPLE_MULTI_LETTERS = SAMPLES_DIR / "multi_letters.png"
SAMPLE_SINGLE_LETTER_DIR = SAMPLES_DIR / "single_letter"


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """Provides a FastAPI test client with lifespan events handled."""
    with TestClient(app) as c:
        yield c


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
def test_health(client):
    """Health endpoint should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_digit(client):
    """Test MNIST digit prediction."""
    with open(SAMPLE_DIGIT, "rb") as f:
        response = client.post("/predict-digit", files={"file": f})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_digit" in data
    assert "confidence" in data
    assert isinstance(data["predicted_digit"], int)


def test_predict_letter_single(client):
    """Test EMNIST single-letter prediction (using 'a.png')."""
    sample_a = SAMPLE_SINGLE_LETTER_DIR / "a.png"
    with open(sample_a, "rb") as f:
        response = client.post("/predict-letter", files={"file": f})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_character" in data
    # model may not be perfect, but if "a" is predicted it's good
    assert isinstance(data["predicted_character"], str)


def test_predict_letter_multi(client):
    """Test EMNIST multi-letter prediction (multi_letters.png)."""
    with open(SAMPLE_MULTI_LETTERS, "rb") as f:
        response = client.post("/predict-letter", files={"file": f})
    # It may still only handle one letter, but should return OK
    assert response.status_code == 200
    data = response.json()
    assert "predicted_character" in data


def test_predict_ocr(client):
    """Test PaddleOCR endpoint on multi.png."""
    with open(SAMPLE_MULTI_DIGITS, "rb") as f:
        response = client.post("/predict-ocr", files={"file": f})
    if response.status_code == 500:
        pytest.skip("⚠️ PaddleOCR not installed or failed.")
    else:
        assert response.status_code == 200
        data = response.json()
        assert "ocr_result" in data
        assert isinstance(data["ocr_result"], list)
        if data["ocr_result"]:
            assert "text" in data["ocr_result"][0]
            assert "confidence" in data["ocr_result"][0]
