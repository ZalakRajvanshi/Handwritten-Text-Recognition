import { useState } from "react";

function getConfidenceColor(conf) {
  if (conf >= 0.8) return "bg-green-500";
  if (conf >= 0.5) return "bg-yellow-500";
  return "bg-red-500";
}

export default function ResultCard({ result }) {
  const [expanded, setExpanded] = useState(false);

  if (!result) return null;

  // Handle error case
  if (result.error || result.detail) {
    return (
      <div className="bg-red-50 border border-red-300 text-red-800 p-4 rounded-md">
        <strong>Error:</strong> {result.error || result.detail}
      </div>
    );
  }

  // Handle OCR response
  if (result.ocr_result) {
    const items = expanded ? result.ocr_result : result.ocr_result.slice(0, 3);

    return (
      <div className="space-y-3">
        <h3 className="font-semibold text-gray-800">OCR Results:</h3>
        {items.map((item, idx) => {
          const confidence = item.confidence;
          const barColor = getConfidenceColor(confidence);
          return (
            <div
              key={idx}
              className="bg-white shadow p-3 rounded-md border flex justify-between items-center"
            >
              <span className="font-mono">{item.text}</span>
              <div className="flex items-center space-x-2">
                <div className="w-24 bg-gray-200 rounded-full h-2">
                  <div
                    className={`${barColor} h-2 rounded-full`}
                    style={{ width: `${(confidence * 100).toFixed(0)}%` }}
                  ></div>
                </div>
                <span className="text-sm text-gray-600">
                  {(confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}

        {result.ocr_result.length > 3 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-blue-600 hover:underline text-sm font-medium"
          >
            {expanded ? "Show Less ▲" : "Show More ▼"}
          </button>
        )}
      </div>
    );
  }

  // Handle digit prediction
  if (result.predicted_digit !== undefined) {
    const confidence = result.confidence;
    const barColor = getConfidenceColor(confidence);

    return (
      <div className="bg-white shadow p-6 rounded-md border text-center">
        <h3 className="font-semibold text-gray-800 mb-3">Digit Prediction</h3>
        <p className="text-4xl font-bold text-blue-700">
          {result.predicted_digit}
        </p>
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`${barColor} h-2 rounded-full`}
              style={{ width: `${(confidence * 100).toFixed(0)}%` }}
            ></div>
          </div>
          <p className="text-sm text-gray-600 mt-1">
            Confidence: {(confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>
    );
  }

  // Handle letter prediction
  if (result.predicted_character) {
    const confidence = result.confidence;
    const barColor = getConfidenceColor(confidence);

    return (
      <div className="bg-white shadow p-6 rounded-md border text-center">
        <h3 className="font-semibold text-gray-800 mb-3">Letter Prediction</h3>
        <p className="text-4xl font-bold text-purple-700">
          {result.predicted_character}
        </p>
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`${barColor} h-2 rounded-full`}
              style={{ width: `${(confidence * 100).toFixed(0)}%` }}
            ></div>
          </div>
          <p className="text-sm text-gray-600 mt-1">
            Confidence: {(confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>
    );
  }

  return null;
}
