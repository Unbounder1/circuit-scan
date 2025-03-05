import cv2
from matplotlib import pyplot as plt

img = cv2.imread("image5.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_size = 7
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Display the grayscale image
plt.imshow(blur_gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()