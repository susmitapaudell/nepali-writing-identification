import cv2

# Load image (by default OpenCV loads in BGR format)
img = cv2.imread("s1.jpg")

#print("Shape of image:", img.shape)   # (height, width, channels)
print("Pixel matrix:\n", img)
