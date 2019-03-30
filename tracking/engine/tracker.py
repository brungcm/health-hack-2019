# -*- coding: utf-8 -*-

import cv2
import logging
import numpy as np

from datetime import datetime
from engine.base_object_estimator import BaseObjectEstimator, TrackingObject
from scipy.spatial.distance import euclidean

logging.basicConfig()
logger = logging.getLogger('Object-Detector')
logger.setLevel(logging.INFO)


class OpenCVTracker(BaseObjectEstimator):
    """ Object tracker with OpenCV built-in methods

        This object tracker works according to the following cyclical way:

        D | T | T | T | T | T | D | T | T ....

        D = *Detection*
        T = *Tracking*

        Once detection is computationally expensive, it is not performed on every frame,
        and thus a tracking algorithm (cheaper) is used.

    """

    def __init__(self, **kwargs):
        """ Constructor

            Args:
                ``detector`` (object): An object detector derived from :class:`~engine.base_object_estimator.BaseObjectEstimator`
                ``detection_rate`` (int): How many frames to perform tracking before the detection step
                ``object_life_cycle`` (int): How many cycles (Detect & Track) to keep track of an object, even if detection fail.
        """

        self.objects = list()
        self.object_life_cycle = kwargs.get('object_life_cycle', 5)
        self.detector = kwargs.get('detector', None)
        self.detection_rate = kwargs.get('detection_rate', 10)
        self.tracker_name = kwargs.get('tracker_name', 'csrt')
        self.frame_idx = 0

        self.tracker_list = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }

        assert self.detector is not None, 'Object detector must be defined'
        assert issubclass(type(self.detector), BaseObjectEstimator), 'Object detector must inherit BaseObjectEstimator'
        assert self.detection_rate > 1, 'Detection rate must be greater than 1'
        assert self.object_life_cycle > 0, 'Life of object must be positive'

    def initialize(self):
        """ Initialization. There is nothing to be done here
        """
        pass

    def process(self, frame):
        """ Process frame throught the Detect & Track cycle

            Args:
                ``frame`` (np.array): frame to be processed

        """

        dead_objects = []
        if (self.frame_idx % self.detection_rate) == 0:  # Detect objects
            self.frame_idx = 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_objects = self.detector.process(frame_rgb)
            logger.info('Objects detected: {}'.format(len(detected_objects)))

            # pairing of detector and tracker centroids, update coordinates and add new objects
            dead_objects = self._update_from_bbox(frame, [obj.bbox for obj in detected_objects])

        else:
            logger.info('Tracking {} objects'.format(len(self.objects)))
            self.frame_idx += 1
            self._update_from_tracker(frame)

        return self.get_objects(), dead_objects

    def _update_from_bbox(self, frame, bboxes):
        """ Update bounding boxes for known objects, add new ones, if any, and deallocate the dead ones.

            Args:
                ``frame`` (np.array): frame to be processed
                ``bboxes`` (list): list of bounding boxes coordinates (xmin, ymin, xmax, ymax)
        """

        new_objects = []

        # Decrease life for all current objects
        for tracker in self.objects:
            tracker.life -= 1

        # Update current objects and create new ones given new set of bboxes
        for bbox in bboxes:
            centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            distances = [euclidean(centroid, tracking_obj.centroid)
                         for tracking_obj in self.objects]

            if len(self.objects) > 0:
                nearest_tracking_obj = np.argmin(distances)
                min_dist = distances[nearest_tracking_obj]
            else:
                nearest_tracking_obj = None
                min_dist = np.inf

            # max_distance is given by half of the human height
            max_distance = int((bbox[2] - bbox[0]) / 2)

            if min_dist > max_distance:
                logger.debug('new object')
                start_time = datetime.now()
                new_objects.append(TrackingObject(bbox[0], bbox[1], bbox[2], bbox[3],
                                                  life=self.object_life_cycle, start_time=start_time,
                                                  tracker=self.tracker_list[self.tracker_name]()))

                new_objects[-1].tracker.init(frame, (new_objects[-1].bbox_xmin, new_objects[-1].bbox_ymin,
                                                     new_objects[-1].bbox_xmax - new_objects[-1].bbox_xmin,
                                                     new_objects[-1].bbox_ymax - new_objects[-1].bbox_ymin))

                # delete old tracker
                if nearest_tracking_obj is not None:
                    del self.objects[nearest_tracking_obj]

            else:
                self.objects[nearest_tracking_obj].update_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
                self.objects[nearest_tracking_obj].life += 1

        self.objects += new_objects

        return self._remove_dead_objects()

    def _update_from_tracker(self, frame):
        """ Update bbox from tracking algorithm

            Args:
                ``frame`` (np.array): frame to be processed
        """
        for obj in self.objects:
            success, pos = obj.tracker.update(frame)
            if success:
                obj.update_bbox(int(pos[0]), int(pos[1]), int(pos[0] + pos[2]), int(pos[1] + pos[3]))

    def _remove_dead_objects(self):
        """ Deallocate dead objects to stop tracking them
        """
        dead_objects_idx = np.where(np.array([tracker.life for tracker in self.objects]) == 0)[0]

        dead_objects = []
        if len(dead_objects_idx) > 0:
            logger.info('Deleting {} dead objects'.format(len(dead_objects_idx)))

            end_time = datetime.now()
            for i in sorted(dead_objects_idx, reverse=True):
                offset = (end_time - self.objects[i].start_time).total_seconds()
                dead_objects.append({'timestamp_start': self.objects[i].start_time, 'offset': offset})
                del self.objects[i]
        return dead_objects

    def get_objects(self):
        """ Returns current detected objects
        """
        return self.objects
