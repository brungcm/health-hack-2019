# -*- coding: utf-8 -*-

import cv2
import logging

from stream.base_stream import BaseVideoStream

logging.basicConfig()
logger = logging.getLogger('OpenCV-VideoStream')
logger.setLevel(logging.INFO)


class OpenCVStream(BaseVideoStream):
    
    """ Video streaming using OpenCV 
        It works both with webcams and video files
    """

    def __init__(self, camera_id):
        """ Constructor

            Args:
                ``camera_id`` (int or str): Camera id (int) or video file path (str)
        """
        self.camera_id = camera_id
        self.camera = None

    def initialize(self):
        """ Open OpenCV camera
        """
        self.camera = cv2.VideoCapture(self.camera_id)

    def next_frame(self):
        """ Returns next frame as BGR numpy array
        """

        ret, frame = self.camera.read()
        if not ret:
            logger.info('Streaming finished')
            self.close()
            return None
        else:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            return frame

    def close(self):
        """ Close camera handler
        """
        self.camera.release()
