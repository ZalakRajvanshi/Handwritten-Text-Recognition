import cv2
import numpy as np

# Create blank white image
img = np.ones((200, 200), dtype=np.uint8) * 255

# Draw digit 7
cv2.putText(img, "7", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,), 10)

cv2.imwrite("samples/digit.png", img)
print("Sample digit image saved at samples/digit.png")
