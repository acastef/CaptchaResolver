import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpi

IMAGE_PATH = '../../data/tdakwq.png'

img = cv2.imread(IMAGE_PATH, 0)

# From RGB to BW
# Otsu threshold with Gaussian Blur
blur = cv2.GaussianBlur(img, (5, 5), 0)
th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
final = cv2.dilate(closing, np.ones((2, 1), np.uint8), iterations=1)

# Plot
titles = ['Original', 'Gaussian + Otsu', 'Closing x 2', 'Dilation']
images = [mpi.imread(IMAGE_PATH), th, closing, final]
for i in range(4):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
