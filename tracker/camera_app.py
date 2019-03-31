import config
import cv2
import json
import logging
import numpy as np
import os
import time

from collections import deque
from engine.classifier import MovementClassifier
from engine.detector import Detector
from engine.face_detector import FaceDetector
from stream.opencv_stream import OpenCVStream

logging.basicConfig()
logger = logging.getLogger('People-Tracker-App')
logger.setLevel(logging.INFO)

FRAME_SIZE = (600, 800, 3)
RESIZE_RATE = 3

POOL_MASK = np.zeros(FRAME_SIZE)
POOL_BORDER_W = int(FRAME_SIZE[1]*0.3)
POOL_SECTOR_W = int(FRAME_SIZE[1]*0.1)
POOL_MASK[:, :POOL_BORDER_W, 0] = 125  # BLUE
POOL_MASK[:, :POOL_BORDER_W, 1] = 242  # GREEN
POOL_MASK[:, :POOL_BORDER_W, 2] = 145  # RED

POOL_MASK[:, POOL_BORDER_W:2*POOL_BORDER_W, 0] = 96   # BLUE
POOL_MASK[:, POOL_BORDER_W:2*POOL_BORDER_W, 1] = 247  # GREEN
POOL_MASK[:, POOL_BORDER_W:2*POOL_BORDER_W, 2] = 242  # RED

POOL_MASK[:, 2*POOL_BORDER_W:, 0] = 96   # BLUE
POOL_MASK[:, 2*POOL_BORDER_W:, 1] = 109  # GREEN
POOL_MASK[:, 2*POOL_BORDER_W:, 2] = 247  # RED


class CameraApp:
    """ App
    """

    def __init__(self, video_stream, object_estimator,face_detector):
        self.video_stream = video_stream
        self.object_estimator=object_estimator
        self.face_detector = face_detector
        self.status = {}

    def run(self):
        """ Run App

            Args:
                ``display`` (bool): Whether or not
        """

        self.video_stream.initialize()
        frame_buffer = deque(maxlen=config.FRAME_BUFFER_SIZE)

        while True:
            frame = self.video_stream.next_frame()
            if frame is None:
                break
            if not frame_buffer:
                frame_buffer = deque([frame] * config.FRAME_BUFFER_SIZE, maxlen=config.FRAME_BUFFER_SIZE)
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_SIZE[1], FRAME_SIZE[0]))            
            frame_buffer.append(frame)
            self.object_estimator.process(frame)
            panic_score = self.movement_classifier.process(frame_buffer)
            self.update_status(panic_score)
            frame = self._draw_bbox(frame, [obj.bbox for obj in self.object_estimator.objects])

            # detect face
            frame, secs = self.face_detector.process(frame)
            print(secs)

            self.write_label(frame)
            if config.DISPLAY:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = (0.6 * frame + 0.4 * POOL_MASK).astype(np.uint8)
                cv2.imshow('Human-Tracking-CameraApp', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

    def update_status(self, panic_score):
        main_obj, cnt = self.get_main_bbox()
        if main_obj:
            self.status['riskPanic'] = panic_score
            self.get_main_object_sector(main_obj)

            self.write_status_to_file()

    def get_main_bbox(self):
        cur_area = 0
        main_object = None
        contour = None
        for obj in self.object_estimator.objects:
            if obj.bbox:
                xmin, ymin, xmax, ymax = obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]
                contour = np.array([[[xmin, ymin]], [[xmax, ymin]], [[xmax, ymax]], [[xmin, ymax]]])
                area = cv2.contourArea(contour)
                if area > cur_area:
                    main_object = obj
                    cur_area = area
        self.status['subject'] = {}
        if main_object:
            self.status['subject'] = {
                'xmin': main_object.bbox[0],
                'xmax': main_object.bbox[2],
                'ymin': main_object.bbox[1],
                'ymax': main_object.bbox[3]
            }
        return main_object, contour

    def get_main_object_sector(self, main_object):
        cx = (main_object.bbox[2] - main_object.bbox[0])/2 + main_object.bbox[0]
        obj_sector = int(cx/POOL_SECTOR_W) * 10
        self.status['riskPosition'] = obj_sector
        self.status['cx'] = cx
        return obj_sector

    def write_label(self, frame):
        if self.status.get('subject', None) and self.status.get('cx', '') and self.status.get('sector', ''):
            xmin_label = 'xmin: {}'.format(self.status.get('subject').get('xmin'))
            xmax_label = 'xmax: {}'.format(self.status.get('subject').get('xmax'))
            cx_label = 'cx: {}'.format(self.status.get('cx', ''))
            sector_label = 'sector: {}'.format(self.status.get('sector', ''))
            cv2.putText(frame, xmin_label, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255),
                        lineType=cv2.LINE_AA)
            cv2.putText(frame, xmax_label, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255),
                        lineType=cv2.LINE_AA)
            cv2.putText(frame, cx_label, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255),
                        lineType=cv2.LINE_AA)
            cv2.putText(frame, sector_label, (20, 65), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255),
                        lineType=cv2.LINE_AA)

    def write_status_to_file(self):
        self.status['updatedAt'] = int(round(time.time() * 1000))
        output_path = os.getenv('STATUS_FILE', '../status.json')
        with open(output_path, 'w') as outfile:
            json.dump(self.status, outfile)

    @staticmethod
    def _draw_bbox(frame, bboxes):
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
            rect_frame = cv2.rectangle(rect_frame, pt1, pt2, (34, 34, 178), thickness=2)
        return rect_frame


def start():
    video_stream = OpenCVStream(config.CAMERA_ID)
    detector = Detector(model_path=config.TRACKER_MODEL_PATH, id2name=config.ID_TO_NAME, threshold=0.5)
    movement_classifier = MovementClassifier(model_path=config.MOVEMENT_CLASSIFIER_MODEL_PATH, threshold=0.5)
    app = CameraApp(video_stream=video_stream, object_estimator=detector, movement_classifier=movement_classifier)
    app.run()
