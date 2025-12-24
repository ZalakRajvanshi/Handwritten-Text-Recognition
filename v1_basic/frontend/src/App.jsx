import { useState } from "react";
import UploadForm from "./components/UploadForm";

function App() {
  const [endpoint, setEndpoint] = useState("predict-ocr");

  return (
    // Full page container with dashed border and padding
    <div className="w-screen h-screen border-4 border-dashed border-blue-500 flex flex-col items-center justify-center bg-gray-50 px-4">
      <div className="text-center mb-8">
        <h1 className="text-3xl sm:text-4xl font-bold text-gray-800">
          Handwritten Recognition AI ✍️
        </h1>
        <div className="flex justify-center space-x-2 mt-4">
          <button
            onClick={() => setEndpoint("predict-digit")}
            className={`px-4 py-2 rounded ${
              endpoint === "predict-digit"
                ? "bg-blue-600 text-white"
                : "bg-white border border-gray-300 text-gray-700"
            }`}
          >
            Digit Recognition
          </button>
          <button
            onClick={() => setEndpoint("predict-letter")}
            className={`px-4 py-2 rounded ${
              endpoint === "predict-letter"
                ? "bg-blue-600 text-white"
                : "bg-white border border-gray-300 text-gray-700"
            }`}
          >
            Letter Recognition
          </button>
          <button
            onClick={() => setEndpoint("predict-ocr")}
            className={`px-4 py-2 rounded ${
              endpoint === "predict-ocr"
                ? "bg-blue-600 text-white"
                : "bg-white border border-gray-300 text-gray-700"
            }`}
          >
            OCR
          </button>
        </div>
      </div>

      <div className="w-full max-w-3xl">
        <UploadForm endpoint={endpoint} />
      </div>
    </div>
  );
}

export default App;
