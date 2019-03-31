import os


CAMERA_ID = os.getenv('CAMERA_ID', 1)
DETECTION_RATE = os.getenv('DETECTION_RATE', 3)
DISPLAY = os.getenv('DISPLAY', 1)
ID_TO_NAME = {1: 'person'}
MODEL_PATH = os.getenv('MODEL_PATH', 'model/ssd_mobilenet_v1_coco_2018_01_28/saved_model')
OBJECT_LIFECYCLE = os.getenv('OBJECT_LIFECYCLE', 3)
TRACKER_NAME = os.getenv('TRACKER_NAME', 'kcf')
WINDOW_SIZE_SECS = os.getenv('WINDOW_SIZE_SECS', 60)
FRAME_BUFFER_SIZE = os.getenv('FRAME_BUFFER_SIZE', 16)
