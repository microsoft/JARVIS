from os.path import join
import numpy as np
from constants import *
import cv2
from sklearn.model_selection import train_test_split


class DatasetLoader:
    def load_from_save(self):
        images = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_IMAGES_FILENAME))
        images = images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        labels = np.load(join(SAVE_DIRECTORY, SAVE_DATASET_LABELS_FILENAME)).reshape([-1, len(EMOTIONS)])
        self._images, self._images_test, self._labels, self._labels_test = train_test_split(images, labels, test_size=0.20, random_state=42)
    
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels

    @property
    def images_test(self):
        return self._images_test

    @property
    def labels_test(self):
        return self._labels_test
