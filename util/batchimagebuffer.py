import numpy as np
import cv2
from enum import Enum
import random


class BatchImageBuffer:
    def __init__(self, num_labels, target_size=0):
        self._paths = []
        self._images = []
        self._labels = []
        self._category_bundle = []

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

    def get_category_bundle(self):
        return self._category_bundle

    def add_categories(self, categories):
        self._category_bundle.append(categories)

    def add_reshaped_image_to_buffer(self, path, image, label):
        if image.shape != self._target_shape:
            raise Exception('Shape of image is not same as buffer\'s shape')
        image = np.reshape(image, image.size)
        label = self._label_to_one_hot_encoding(label)

        self._paths.append(path)
        self._images.append(image)
        self._labels.append(label)

    def shape(self):
        return self._target_shape

    def num_label(self):
        return self._num_labels

    def shuffle(self):
        buffer = list(zip(self._images, self._labels, self._paths))
        random.shuffle(buffer)
        self._images, self._labels, self._paths = zip(*buffer)

    def reset(self):
        self._current_index = 0

    def get_images(self):
        return self._images

    def get_labels(self):
        return self._labels

    def size(self):
        return len(self._images)

    def next_batch(self, batch_size):
        start_index = self._current_index
        end_index = self._current_index+batch_size
        self._current_index = end_index
        if end_index > self.size():
            end_index = self.size()
            self.reset()
        return self._images[start_index:end_index], self._labels[start_index:end_index], self._paths[
                                                                                            start_index:end_index]
