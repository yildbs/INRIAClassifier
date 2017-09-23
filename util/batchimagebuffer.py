import numpy as np
import cv2
from enum import Enum
import random


class BatchImageBuffer:
    def __init__(self, num_labels, target_size=0):
        self._buffer = [] # list of [filename, image, label]
        if target_size == 0:
            target_size = [128, 128] # Default image size is 128x128
        self._target_size = target_size
        self._target_shape = tuple([target_size[0], target_size[1], 3])
        self._current_index = 0
        self._num_labels = num_labels

    def _label_to_one_hot_encoding(self, label):
        one_hot_array = np.zeros((self._num_labels))
        one_hot_array[label] = 1
        return one_hot_array

    def add_reshaped_image_to_buffer(self, path, image, label):
        if image.shape != self._target_shape:
            raise Exception('Shape of image is not same as buffer\'s shape')

        # TODO. Check
        image = np.reshape(image, image.size)
        label = self._label_to_one_hot_encoding(label)
        self._buffer.append([image, label, path])

    def shape(self):
        return self._target_shape

    def num_label(self):
        return self._num_labels

    def shuffle(self):
        random.shuffle(self._buffer)

    def reset(self):
        self._current_index = 0

    def images(self):
        return self._buffer[:][1]

    def labels(self):
        return self._buffer[:][2]

    def size(self):
        return len(self._buffer)

    def next_batch(self, batch_size):
        start_index = self._current_index
        end_index = self._current_index+batch_size
        if end_index >= len(self._buffer):
            end_index = len(self._buffer)
            self.reset()
        return self._buffer[start_index:end_index]
