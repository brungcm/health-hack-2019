import cv2
import numpy as np
import json
import os

from argparse import ArgumentParser
from video_stream import OpenCVStream
from engine import Detector

from datetime import datetime
from collections import deque

import logging
logging.basicConfig()
logger = logging.getLogger('Human-Counter-ExtractFrames')
logger.setLevel(logging.INFO)

from utils import IMAGE_SIZE

class ExtractFrames:

    def __init__(self, detector, video_stream):
        self.detector = detector
        self.video_stream = video_stream

    def run(self, camera_id, output_dir):
        """ Run App

            Args:
                ``display`` (bool): Whether or not
                ``audit_writer`` (cv2.VideoWriter): Video writer handler
        """

        logger.info('Camera ID: {}'.format(camera_id))
        logger.info('Display frames: {}'.format(display))
        os.makedirs(output_dir)

        frame_count = 0
        while True:

            frame = self.video_stream.next_frame()
            if frame is None:
                break
            logger.info(frame.shape)
            objects = self.detector.process(frame)
            
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            
            logger.info(objects)

            for obj_i, obj in enumerate(objects):
                frame_cropped = frame[obj.bbox_ymin:obj.bbox_ymax,
                                      obj.bbox_xmin:obj.bbox_xmax]
                output_image_path = os.path.join(
                    output_dir, '{}_frame_{}_obj_{}.jpg'.format(os.path.basename(camera_id), frame_count, obj_i))
                logger.info(output_image_path)
                frame_resize = cv2.resize(frame_cropped,IMAGE_SIZE)
                cv2.imwrite(output_image_path, frame_resize)
            frame_count += 1


if __name__ == '__main__':

    parser = ArgumentParser(prog="Extract Frames")
    parser.add_argument('--config-file', required=True,
                        dest='config_file', type=str)
    parser.add_argument('--video-path', required=True,
                        dest='video_path', type=str)
    parser.add_argument('--output-dir', required=True,
                        dest='output_dir', type=str)
    args = parser.parse_args()

    config = json.load(open(args.config_file, 'r'))

    model_path = config['object_detection']['model_path']
    id2name = {int(k): v for k,
               v in config['object_detection']['id2name'].items()}

    camera_id = args.video_path

    display = False

    detector = Detector(model_path=model_path, id2name=id2name, threshold=0.8)

    stream = OpenCVStream(camera_id)
    stream.initialize()

    app = ExtractFrames(detector, stream)
    app.run(camera_id, args.output_dir)
