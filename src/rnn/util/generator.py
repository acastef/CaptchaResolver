import os
import random

import numpy as np
from keras import backend as K

from .captcha_processor import CAPTCHAProcessor

# A-Z 0-9
LETTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', '_'
]


def text_to_labels(text):
    return list(map(lambda x: LETTERS.index(x), text))


def is_valid_str(s):
    for ch in s:
        if ch not in LETTERS:
            return False
    return True


class ImageGenerator:

    def __init__(self, dirpath, img_w, img_h, batch_size,
                 downsample_factor, max_text_len=8):

        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.imgs = None
        self.texts = None

        self.samples = []
        for filename in os.listdir(dirpath):
            name, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg']:
                img_filepath = os.path.join(dirpath, filename)
                ann = name.upper()
                if is_valid_str(ann):
                    self.samples.append([img_filepath, ann])

        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0

    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            image = CAPTCHAProcessor.process_captcha(img_filepath, self.img_w, self.img_h)
            image = image.astype(np.float32)
            image /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = image
            for j in range(self.max_text_len - len(text)):
                text += '_'
            self.texts.append(text)

    @staticmethod
    def get_output_size():
        return len(LETTERS) + 1

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []

            for i in range(self.batch_size):
                image, text = self.next_sample()
                image = image.T
                if K.image_data_format() == 'channels_first':
                    image = np.expand_dims(image, 0)
                else:
                    image = np.expand_dims(image, -1)
                X_data[i] = image
                Y_data[i] = text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
