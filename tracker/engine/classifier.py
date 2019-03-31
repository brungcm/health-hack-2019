# -*- coding: utf-8 -*-

import cv2
import io
import tensorflow as tf

from PIL import Image
from scipy.special import softmax

IMAGE_SIZE = (300, 300)


class MovementClassifier(object):

    """ Movement classifier using trained model
    """

    def __init__(self, **kwargs):
        """
            Args:
                ``kwargs`` (dict): Detector arguments. They are:
                ``model_path`` (str): model path to saved model
                ``threshold`` (float): Detection threshold
        """

        self.model = None
        self.model_path = kwargs.get('model_path', None)
        self.threshold = kwargs.get('threshold', 0.5)
        self.objects = None
        assert self.model_path is not None, 'Path of detection model must be set'

    def initialize(self):
        self.model = tf.contrib.predictor.from_saved_model(self.model_path)

    def process(self, frame_buffer=[]):
        if self.model is None:
            self.initialize()
        frames = {}
        list(frame_buffer)
        for idx, f in enumerate(list(frame_buffer)):
            resized = cv2.resize(f, IMAGE_SIZE)
            key = 'frame_{}'.format(idx)
            img = Image.fromarray(resized)
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format='JPEG')
            frames[key] = [img_byte_array.getvalue()]
        predict = self.model(frames)
        panic_score = softmax(predict.get('output'))[1] * 100
        return panic_score
