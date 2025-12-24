import { useState } from "react";
import UploadForm from "./UploadForm";

const tabs = [
  { id: "predict-digit", label: "Digit Recognition" },
  { id: "predict-letter", label: "Letter Recognition" },
  { id: "predict-ocr", label: "OCR" },
];

export default function Tabs() {
  const [active, setActive] = useState("predict-digit");

  return (
    <div>
      <div className="flex justify-center space-x-4 mb-6">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActive(tab.id)}
            className={`px-4 py-2 rounded-md ${
              active === tab.id
                ? "bg-blue-600 text-white"
                : "bg-gray-200 hover:bg-gray-300"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <UploadForm endpoint={active} />
    </div>
  );
}
