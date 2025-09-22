import React, { useState, useRef } from "react";
import ResultCard from "./ResultCard";

export default function UploadForm({ endpoint }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
      setResult(null);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    // NEW: File type check
    if (!["image/png", "image/jpeg"].includes(selectedFile.type)) {
    alert("Only PNG and JPG files are supported");
    return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
  };  


  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/${endpoint}`, {
      method: "POST",
      body: formData,
      });

      if (!res.ok) {
      throw new Error(`Server responded with ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Error uploading file:", err);
      setResult({ error: "Failed to get response from server" });
    }
  };

  const onButtonClick = () => {
      inputRef.current.click();
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl mx-auto">
      <input
        ref={inputRef}
        type="file"
        accept="image/png, image/jpeg"
        className="hidden"
        onChange={handleFileChange}
      />

      <div
        onClick={onButtonClick}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        className={`cursor-pointer rounded-md border-2 border-dashed p-20 flex flex-col justify-center items-center
          ${
            dragActive
              ? "border-blue-600 bg-blue-50"
              : "border-gray-300 bg-white hover:border-blue-400"
          }
        `}
      >
        {preview ? (
          <img
            src={preview}
            alt="Preview"
            className="max-h-48 rounded-md object-contain"
          />
        ) : (
          <>
            <p className="text-gray-500 text-lg font-medium">Click or drag image here</p>
            <p className="text-gray-400 text-sm mt-1">PNG, JPG supported</p>
          </>
        )}
      </div>

      <button
        type="submit"
        disabled={loading || !file}
        className={`mt-4 w-full py-3 rounded-md text-white font-semibold transition ${
          loading || !file
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700"
        }`}
      >
        {loading ? "Processing..." : "Predict"}
      </button>

      {result && (
        <div className="mt-6">
          <ResultCard result={result} />
        </div>
      )}
    </form>
  );
}