import cv2
import numpy as np


class CAPTCHAProcessor:

    @staticmethod
    def process_captcha(captcha, width=None, height=None):
        image = cv2.imread(captcha, 0)

        # From RGB to BW
        # Adaptive thresholding
        th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

        dilation = cv2.dilate(th, np.ones((4, 2), np.uint8), iterations=1)
        erosion = cv2.erode(dilation, np.ones((1, 2), np.uint8), iterations=1)
        final = cv2.dilate(erosion, np.ones((1, 2), np.uint8), iterations=1)

        if width and height:
            final = cv2.resize(final, (width, height))
        return final
