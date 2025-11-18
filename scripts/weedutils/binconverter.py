import cv2
import numpy as np

# 1. Read the image using OpenCV
image = cv2.imread("/home/vjti-comp/Downloads/WEEDRICE_RGB.JPG")  # Load the image (BGR format by default)

# 2. Optionally, convert to RGB if needed
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Convert to a NumPy tensor (array)
tensor = np.array(image_rgb, dtype=np.float32)  # or np.uint8 if you prefer

# 4. Save tensor to a .bin file
np.save("/home/vjti-comp/Downloads/testoutput.npy", tensor)

print("Image converted to tensor and saved as 'output.npy'.")
