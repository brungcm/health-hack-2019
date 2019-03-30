# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from engine.base_object_estimator import BaseObjectEstimator, TrackingObject


class Detector(BaseObjectEstimator):
    """ Object detector using `Tensorflow Object Detection API <https://github.com/tensorflow/models/tree/master/research/object_detection>`_

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

    def process(self, frame):
        if self.model is None:
            self.initialize()

        height, width = frame.shape[:2]

        predict = self.model({'inputs': [frame]})
        _objects = self._parse_prediction(predict)
        _objects = self._scale_bbox(_objects, height, width)

        self.objects = [TrackingObject(bbox_xmin=obj['detection_bbox_original'][0],
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
        # filter objects by score
        idx_high = np.where(output_dict['detection_scores'][0] >= self.threshold)[0]
        idx_object_ids = output_dict['detection_classes'][0][idx_high].astype(int)

        # filter objects by objects of interest
        idx_filtered_object_ids = np.where(np.isin(idx_object_ids, list(self.id2name.keys())))[0]

        detection_classes = output_dict['detection_classes'][0][idx_filtered_object_ids]
        detection_scores = output_dict['detection_scores'][0][idx_filtered_object_ids]
        detection_boxes = output_dict['detection_boxes'][0][idx_filtered_object_ids]

        parsed_pred = [{'detection_class': class_id, 'detection_score': score, 'detection_bbox': bbox}
                       for class_id, score, bbox in zip(detection_classes, detection_scores, detection_boxes)]
        return parsed_pred

    @staticmethod
    def _scale_bbox(parsed_pred, height, width):
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
        return self.objects
