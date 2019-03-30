# -*- coding: utf-8 -*-
"""
.. module:: engine
   :platform: Unix
   :synopsis: Definition of object detector using `Tensorflow Object Detection API <https://github.com/tensorflow/models/tree/master/research/object_detection>`_ 

.. moduleauthor:: Rodrigo Pereira <rodrigofp@ciandt.com>

"""

from __future__ import division

import tensorflow as tf
import cv2
import numpy as np
import base_object_estimator
import sys

import logging
logging.basicConfig()
logger = logging.getLogger('Object-Detector')
logger.setLevel(logging.INFO)


class Detector(base_object_estimator.BaseObjectEstimator):
# class Detector():
    """ Object detector using `Tensorflow Object Detection API <https://github.com/tensorflow/models/tree/master/research/object_detection>`_

    """

    def __init__(self,model_path,id2name,threshold=0.5):
        """

            Args:
                ``kw`` (dict): Detector arguments. They are:
                    ``model_path`` (str): model path to saved model
                    ``id2name`` (dict): dictionary that maps the object id (int) to object name (str). See config files
                    ``threshold`` (float): Detection threshold
        """

        self.model = None
        self.model_path = model_path
        self.id2name = id2name
        self.threshold = threshold
        self.objects = None

        assert self.model_path is not None, 'Path of detection model must be set'
        assert self.id2name is not None, 'id2name dict must be set'

        self.initialize()

    def initialize(self):
        """ Load Tensorflow saved model
        """
        self.model = tf.contrib.predictor.from_saved_model(self.model_path)

    def process(self, frame):
        """ Detect objects given an image and persist them in-memory
            It is been used the `Tensorflow Object Detection API <https://github.com/tensorflow/models/tree/master/research/object_detection>`_ 

            Args:
                ``frame`` (numpy array): Image
        """

        # if self.model is None:
        #     self.initialize()

        height, width = frame.shape[:2]

        print("predicting")
        predict = self.model({'inputs': [frame]})
        print("predicted")
        _objects = self._parse_prediction(predict)
        _objects = self._scale_bbox(_objects, height, width)

        self.objects = [base_object_estimator.TrackingObject(bbox_xmin=obj['detection_bbox_original'][0],
                                                             bbox_ymin=obj['detection_bbox_original'][1],
                                                             bbox_xmax=obj['detection_bbox_original'][2],
                                                             bbox_ymax=obj['detection_bbox_original'][3],
                                                             life=-1, tracker=None, obj_id=None,
                                                             obj_type_id=obj['detection_class'],
                                                             obj_type_name=self.id2name[obj['detection_class']],
                                                             obj_score=obj['detection_score'])
                        for obj in _objects]
        return self.objects

    def _parse_prediction(self, output_dict):
        """ Parse `Tensorflow Object Detection API <https://github.com/tensorflow/models/tree/master/research/object_detection>`_ output bounding boxes.

            First, filter objects by prediction score and then by objects of interest.

            Args:
                ``output_dict`` (dict): Output predictions by Object Detection API.

            Returns:
                ``parsed_pred`` (dict): Parsed and filtered objects
        """

        # filter objects by score
        idx_high = np.where(
            output_dict['detection_scores'][0] >= self.threshold)[0]
        
        idx_object_ids = output_dict['detection_classes'][0][idx_high]
        

        # filter objects by objects of interest
        idx_filtered_object_ids = np.where(
            np.isin(idx_object_ids, self.id2name.keys()))[0]        

        detection_classes = output_dict['detection_classes'][0][idx_filtered_object_ids]
        detection_scores = output_dict['detection_scores'][0][idx_filtered_object_ids]
        detection_boxes = output_dict['detection_boxes'][0][idx_filtered_object_ids]

        parsed_pred = [{'detection_class': class_id, 'detection_score': score, 'detection_bbox': bbox}
                       for class_id, score, bbox in zip(detection_classes, detection_scores, detection_boxes)]

        return parsed_pred

    def _scale_bbox(self, parsed_pred, height, width):
        """ Rescale bounding boxes to original image dimensions

            Args:
                ``parsed_pred`` (dict): Parsed and filtered objects
                ``height`` (dict): Height of original image
                ``width`` (dict): Width of original image

            Returns:
                ``parsed_pred`` (dict): Parsed and filtered objects with bouding boxes rescaled to original image dimensions
        """

        for obj in parsed_pred:
            obj['detection_bbox_original'] = [
                int(width * obj['detection_bbox'][1]),  # x_min
                int(height * obj['detection_bbox'][0]),  # y_min
                int(width * obj['detection_bbox'][3]),  # x_max
                int(height * obj['detection_bbox'][2])  # y_max
            ]

            obj['centroid'] = [int((obj['detection_bbox_original'][0] + obj['detection_bbox_original'][2]) / 2),
                               int((obj['detection_bbox_original'][1] + obj['detection_bbox_original'][3]) / 2)]

        return parsed_pred

    def get_objects(self):
        """ Return all current objects
        """
        return self.objects


if __name__ == '__main__':

    image = cv2.imread('../images/people_sample1.jpg')
    model_path = '../pretrained-models/ssd_mobilenet_v1_coco_2018_01_28/saved_model'

    # TODO: convert labels file (Protobuf) to dict+
    id2name = {
        1: 'person',
        # 72: 'tv',
        # 73: 'laptop',
        # 74: 'mouse',
        # 76: 'keyboard',
        # 77: 'mobile phone'
    }

    detector = Detector(model_path, id2name)
    detector.process(image)
    logger.info(detector.get_objects())
