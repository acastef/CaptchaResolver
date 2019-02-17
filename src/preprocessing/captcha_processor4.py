import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpi

IMAGE_PATH = '../../data/1c128a57.png'

img = cv2.imread(IMAGE_PATH, 0)

# From RGB to BW
# Adaptive thresholding
th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

dilation = cv2.dilate(th, np.ones((4, 2), np.uint8), iterations=1)
erosion = cv2.erode(dilation, np.ones((1, 2), np.uint8), iterations=1)
final = cv2.dilate(erosion, np.ones((1, 2), np.uint8), iterations=1)

# Plot
titles = ['Original', 'Adaptive + Gaussian + Otsu', 'Dilation', 'Opening']
images = [mpi.imread(IMAGE_PATH), th, dilation, final]
for i in range(4):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
