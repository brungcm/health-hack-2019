import config
import cv2
import logging
import numpy as np
import time

from copy import copy
from datetime import datetime
from engine.base_object_estimator import BaseObjectEstimator
from engine.detector import Detector
from engine.tracker import OpenCVTracker
from stream.base_stream import BaseVideoStream
from stream.opencv_stream import OpenCVStream

logging.basicConfig()
logger = logging.getLogger('People-Tracker-App')
logger.setLevel(logging.INFO)

RESIZE_RATE = 3


class CameraApp:
    """ App
    """

    def __init__(self, video_stream, object_estimator, window_size_secs=60):

        """ Constructor

            Args:
                ``video_stream`` (object):
                ``object_estimator`` (object):
                ``window_size_secs``: Time window to make estimation
        """

        assert video_stream is not None, 'Video Streaming must be provided'
        assert issubclass(type(video_stream), BaseVideoStream), 'Video streamer must be subclass of BaseVideoStream'
        assert object_estimator is not None, 'Object estimator must be provided'
        assert issubclass(type(object_estimator),
                          BaseObjectEstimator), 'Object estimator must be subclass of BaseObjectEstimator'
        assert window_size_secs > 0, 'window_size_secs must be positive'

        self.video_stream = video_stream
        self.object_estimator = object_estimator
        self.window_size_secs = window_size_secs

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

    def run(self, display=False, audit_writer=None):
        """ Run App

            Args:
                ``display`` (bool): Whether or not
                ``audit_writer`` (cv2.VideoWriter): Video writer handler
        """

        obj_count_queue = list()
        time_start = datetime.now()
        self.video_stream.initialize()
        dead_objects = []

        while True:
            frame = self.video_stream.next_frame()
            if frame is None:
                break
            objs, _dead_objects = self.object_estimator.process(frame)
            dead_objects += _dead_objects
            obj_count_queue.append(len(objs))
            time_end = datetime.now()

            if (time_end - time_start).total_seconds() > self.window_size_secs:
                mean_objects = np.ceil(np.mean(obj_count_queue))
                first_quantile, second_quantile, third_quantile = np.percentile(obj_count_queue, [25, 50, 75])

                logger.info('Objects from {} - {}  =  mean: {}, min: {}, median: {}, max: {}'.format(
                    time_start, time_end, mean_objects, first_quantile, second_quantile, third_quantile))

                del obj_count_queue
                obj_count_queue = list()
                del dead_objects
                dead_objects = []
                time_start = time_end

            frame = self._draw_bbox(
                frame, [obj.bbox for obj in self.object_estimator.objects])

            if display:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Human-Tracking-CameraApp', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if audit_writer is not None:
                w = self.video_stream.get(cv2.CV_CAP_PROP_FRAME_WIDTH)
                h = self.video_stream.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)
                frame_audit = cv2.resize(frame, (int(w / RESIZE_RATE), int(h / RESIZE_RATE)))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame_audit, 'max-distance: {}'.format(self.tracker.max_distance),
                            (15, 25), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame_audit, 'skip-frames: {}'.format(self.skip_frames), (15, 50), font,
                            0.6, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame_audit, 'num: {}'.format(len(self.tracker.trackers)), (15, 75), font,
                            0.6, (255, 0, 0), 1, cv2.LINE_AA)
                audit_writer.write(frame_audit)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = Detector(model_path=config.MODEL_PATH, id2name=config.ID_TO_NAME, threshold=0.5)
    tracker = OpenCVTracker(detector=detector, detection_rate=config.DETECTION_RATE,
                            object_life_cycle=config.OBJECT_LIFECYCLE, tracker_name=config.TRACKER_NAME)
    opencv_stream = OpenCVStream(config.CAMERA_ID)
    audit_writer = None
    app = CameraApp(video_stream=opencv_stream, object_estimator=tracker, window_size_secs=config.WINDOW_SIZE_SECS)
    app.run(display=config.DISPLAY, audit_writer=audit_writer)
