from infer import HandwritingRecognizer

ocr = HandwritingRecognizer(
    "crnn_line_v2.pth",
    "charset.json"
)

text = ocr.recognize("test_line.jpeg")
print("Recognized:", text)
