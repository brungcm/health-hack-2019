# -*- coding: utf-8 -*-
"""
.. module:: apps
   :platform: Unix
   :synopsis: App for estimate people density using Detect & Track approach

.. moduleauthor:: Rodrigo Pereira <rodrigofp@ciandt.com>

"""

import cv2
import numpy as np

import json
from copy import copy

from video_stream.base_stream import BaseVideoStream
from video_stream.opencv_stream import OpenCVStream
from engine import detector, tracker
from engine.base_object_estimator import BaseObjectEstimator

from datetime import datetime
from argparse import ArgumentParser


import logging
logging.basicConfig()
logger = logging.getLogger('People-Tracker-App')
logger.setLevel(logging.INFO)

# FRAME_SIZE = (480,360,3)
# FRAME_SIZE = (360,480,3)
FRAME_SIZE = (480,640,3)
POOL_BORDER_W = int(FRAME_SIZE[1]*0.3)

POOL_MASK = np.zeros(FRAME_SIZE)

POOL_MASK[:,:POOL_BORDER_W,0] = 212 # BLUE
POOL_MASK[:,:POOL_BORDER_W,1] = 255 # GREEN
POOL_MASK[:,:POOL_BORDER_W,2] = 127 # RED

POOL_MASK[:,POOL_BORDER_W:2*POOL_BORDER_W,0] = 96 # BLUE
POOL_MASK[:,POOL_BORDER_W:2*POOL_BORDER_W,1] = 247 # GREEN
POOL_MASK[:,POOL_BORDER_W:2*POOL_BORDER_W,2] = 242 # RED

POOL_MASK[:,2*POOL_BORDER_W:,0] = 96 # BLUE
POOL_MASK[:,2*POOL_BORDER_W:,1] = 109 # GREEN
POOL_MASK[:,2*POOL_BORDER_W:,2] = 247 # RED

class CameraApp:
    """ App
    """

    def __init__(self,
                 video_stream,
                 object_estimator,
                 app_logger=None,
                 app_timer_logger=None,
                 window_size_secs=5,
                 pubsub_publisher=None,
                 pubsub_topic=None):
        """ Constructor

            Args:
                ``video_stream`` (object): 
                ``object_estimator`` (object): 
                ``app_logger`` (:class:`~utils.BigQueryConn`): BigQuery Connector 
                ``window_size_secs``: Time window to make estimation
        """

        assert video_stream is not None, "Video Streaming must be provided"
        assert issubclass(type(
            video_stream), BaseVideoStream), "Video streamer must be subclass of BaseVideoStream"

        assert object_estimator is not None, "Object estimator must be provided"
        assert issubclass(type(
            object_estimator), BaseObjectEstimator), "Object estimator must be subclass of BaseObjectEstimator"

        assert window_size_secs > 0, "window_size_secs must be positive"

        self.video_stream = video_stream
        self.object_estimator = object_estimator
        self.window_size_secs = window_size_secs        

    def _draw_bbox(self, frame, bboxes):
        """ Draw bounding boxes on current frame

            Args:

                ``frame`` (numpy array): image
                ``bboxes`` (list(tuple)): bboxes coordinates in the form (xmin,xmax,ymin,ymax)

            Returns:
                Image (numpy array) with bouding boxes drawn
        """

        rect_frame = frame
        for bbox in bboxes:

            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[2], bbox[3])
            rect_frame = cv2.rectangle(
                rect_frame, pt1, pt2, (34, 34, 178), thickness=2)  # TODO: auto-color

        return rect_frame

    def run(self, display=False, audit_writer=None):
        """ Run App

            Args:
                ``display`` (bool): Whether or not
                ``audit_writer`` (cv2.VideoWriter): Video writer handler
        """

        obj_count_queue = list()
        # time_start = datetime.now()

        self.video_stream.initialize()        

        while True:

            frame = self.video_stream.next_frame()
            if frame is None:
                break            
            frame = cv2.resize(frame,(FRAME_SIZE[1],FRAME_SIZE[0]))
            
            objs = self.object_estimator.process(frame)
                        
            frame = self._draw_bbox(
                frame, [obj.bbox for obj in self.object_estimator.objects])

            if display:
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                frame = (0.6*frame + 0.4*POOL_MASK).astype(np.uint8)                

                cv2.imshow('Human-Tracking-CameraApp', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = ArgumentParser(prog="People Tracker")
    parser.add_argument('--config-file', required=True,
                        dest='config_file', type=str)
    args = parser.parse_args()

    config = json.load(open(args.config_file, 'r'))

    model_path = config['object_detection']['model_path']
    object_life_cycle = config['object_detection']['object_life_cycle']
    threshold = float(config['object_detection']['threshold'])
    detection_rate = int(config['object_detection']['detection_rate'])
    id2name = {int(k): v for k,
               v in config['object_detection']['id2name'].items()}    

    window_size_secs = config['window_size_secs']

    try:
        camera_id = int(config['video_stream']['camera_id'])
    except ValueError as err:
        camera_id = config['video_stream']['camera_id']
    

    audit_writer = None
    display = True

    detector = detector.Detector(
        model_path=model_path, id2name=id2name, threshold=threshold)

    opencv_stream = OpenCVStream(camera_id)

    app = CameraApp(video_stream=opencv_stream,
                    object_estimator=detector)
    app.run(display=display)
