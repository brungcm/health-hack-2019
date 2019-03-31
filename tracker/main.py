# coding=utf-8

import camera_app
import logging

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
    camera_app.start()
