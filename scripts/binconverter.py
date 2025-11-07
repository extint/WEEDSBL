import cv2
import numpy as np

# 1. Read the image using OpenCV
image = cv2.imread("/home/vjti-comp/Downloads/drive-download-20251106T100343Z-1-001/1_jpg.rf.0f181860d5708c83d244220fbea051b6.jpg")  # Load the image (BGR format by default)

# 2. Optionally, convert to RGB if needed
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Convert to a NumPy tensor (array)
tensor = np.array(image_rgb, dtype=np.float32)  # or np.uint8 if you prefer

# 4. Save tensor to a .bin file
tensor.tofile("/home/vjti-comp/Downloads/testoutput.bin")

print("Image converted to tensor and saved as 'output.bin'.")
