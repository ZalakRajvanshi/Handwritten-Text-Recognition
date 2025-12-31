import React, { useState } from "react";
import "./styles.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;

    // Validate file type
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file");
      return;
    }

    // Validate file size (max 5MB)
    if (f.size > 5 * 1024 * 1024) {
      setError("File size must be less than 5MB");
      return;
    }

    setFile(f);
    setPreview(URL.createObjectURL(f));
    setText("");
    setError("");
  };

  const recognize = async () => {
    if (!file) return;

    setLoading(true);
    setError("");

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch("http://127.0.0.1:8000/ocr", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      setText(data.text || "No text recognized");
    } catch (err) {
      setError(err.message || "Failed to recognize text");
      setText("");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setText("");
    setError("");
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="page">
      {/* HEADER */}
      <header className="header">
        <div className="header-inner">
          <h1>Handwritten OCR</h1>
          <p>
            CRNN-based single-line handwriting recognition — offline & fast
          </p>
        </div>
      </header>

      {/* MAIN */}
      <main className="main">
        <div className="container">
          <div className="card">
            {/* Upload Section */}
            <div
              className="upload"
              onClick={() => document.getElementById("fileInput").click()}
              role="button"
              tabIndex={0}
              onKeyPress={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  document.getElementById("fileInput").click();
                }
              }}
            >
              <div className="upload-title">
                Upload a handwritten line image
              </div>
              <div className="upload-sub">
                Single-line text • JPG / PNG • Clear handwriting
              </div>

              <input
                id="fileInput"
                type="file"
                accept="image/*"
                onChange={handleFile}
                aria-label="Upload image file"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            {/* Image Preview */}
            {preview && (
              <img src={preview} alt="Uploaded preview" className="preview" />
            )}

            {/* Action Buttons */}
            <div className="button-group">
              <button
                className="button button-primary"
                onClick={recognize}
                disabled={!file || loading}
                aria-busy={loading}
              >
                {loading ? "Recognizing…" : "Recognize Text"}
              </button>
              {file && (
                <button
                  className="button button-secondary"
                  onClick={resetForm}
                >
                  Reset
                </button>
              )}
            </div>

            {/* Output Section */}
            {text && (
              <div className="output">
                <div className="output-header">
                  <div className="output-label">Recognized Text</div>
                  <div className="badge">CRNN Output</div>
                </div>
                <div className="output-text">{text}</div>
                <button
                  className="button button-copy"
                  onClick={copyToClipboard}
                  title="Copy to clipboard"
                >
                  Copy
                </button>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* FOOTER */}
      <footer className="footer">
        v2 — Single-line handwriting OCR • No language model • Offline inference
      </footer>
    </div>
  );
}

export default App;