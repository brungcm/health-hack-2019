# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time


class FaceDetector(object):

    """ Haar cascade frontal face detector
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            '/tracker/tracker/model/haarcascade_frontalface_default.xml')

        self.start_flag=False
        self.start = 0
        self.tolerance_frames = 10
        self.tolerance_counter = 0
        self.duration = 0

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0 and not self.start_flag:
            self.start_flag=True
            self.start = time.time()
            self.duration = 0 
            self.tolerance_counter=0
        elif len(faces) > 0 and self.start_flag:
            self.duration = time.time() - self.start
        elif len(faces) == 0 and self.tolerance_counter < self.tolerance_frames:
            self.tolerance_counter+=1
            # self.duration = time.time() - self.start
        elif len(faces) == 0 and self.tolerance_counter >= self.tolerance_frames:
            self.start_flag=False
            self.start = time.time()
            self.duration = 0 
            self.tolerance_counter=0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        

        return frame, self.duration
