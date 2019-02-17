import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpi

IMAGE_PATH = '../../data/1gd84.png'

img = cv2.imread(IMAGE_PATH, 0)
img = cv2.bitwise_not(img)

# From RGB to BW
# Otsu thresholding
th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

dilation = cv2.dilate(th, np.ones((4, 2), np.uint8), iterations=1)
final = cv2.erode(dilation, np.ones((1, 2), np.uint8), iterations=2)

titles = ['Original', 'Otsu', 'Dilation', 'Erosion']
images = [mpi.imread(IMAGE_PATH), th, dilation, final]

for i in range(4):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
