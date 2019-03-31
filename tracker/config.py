import os


CAMERA_ID = os.getenv('CAMERA_ID', 0)
DETECTION_RATE = os.getenv('DETECTION_RATE', 3)
DISPLAY = os.getenv('DISPLAY', 1)
ID_TO_NAME = {1: 'person'}
TRACKER_MODEL_PATH = os.getenv('TRACKER_MODEL_PATH', 'model/ssd_mobilenet_v1_coco_2018_01_28/saved_model')
MOVEMENT_CLASSIFIER_MODEL_PATH = os.getenv('MOVEMENT_CLASSIFIER_MODEL_PATH', 'model/movement_classifier/saved_model/v1')
OBJECT_LIFECYCLE = os.getenv('OBJECT_LIFECYCLE', 3)
TRACKER_NAME = os.getenv('TRACKER_NAME', 'kcf')
WINDOW_SIZE_SECS = os.getenv('WINDOW_SIZE_SECS', 60)
FRAME_BUFFER_SIZE = os.getenv('FRAME_BUFFER_SIZE', 16)
STATUS_FILE = os.getenv('STATUS_FILE', '/data/status.json')
CASCADE_CLASSIFIER_PATH = os.getenv('CASCADE_CLASSIFIER_PATH', 'model/haarcascade_frontalface_default.xml')
PANIC_THRESHOLD = os.getenv('PANIC_THRESHOLD', 0.04)

