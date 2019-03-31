# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class MovementClassifier(object):

    """ Movement classifier using trained model
    """

    def __init__(self, **kwargs):
        """
            Args:
                ``kwargs`` (dict): Detector arguments. They are:
                ``model_path`` (str): model path to saved model
                ``id2name`` (dict): dictionary that maps the object id (int) to object name (str). See config files
                ``threshold`` (float): Detection threshold
        """

        self.model = None
        self.model_path = kwargs.get('model_path', None)
        self.id2name = kwargs.get('id2name', None)
        self.threshold = kwargs.get('threshold', 0.5)
        self.objects = None
        assert self.model_path is not None, 'Path of detection model must be set'
        assert self.id2name is not None, 'id2name dict must be set'

    def initialize(self):
        self.model = tf.contrib.predictor.from_saved_model(self.model_path)

    def process(self, frame_buffer=[]):
        if self.model is None:
            self.initialize()
        return None
