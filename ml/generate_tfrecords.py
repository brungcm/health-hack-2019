# -*- coding: utf-8 -*-

from __future__ import division

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import datetime

import sys
import os
import argparse
import json

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors

TARGET_DICT = {
    'normal': 0,
    'drowning': 1
}

# TODO: Fazer resize antes de salvar

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir',
                        dest='input_dir',
                        type=str,
                        required=True)
    parser.add_argument('--output-dir',
                        dest='output_dir',
                        type=str,
                        required=True)
    parser.add_argument('--target',
                        dest='target',
                        type=str,
                        choices=['normal', 'drowning'],
                        required=True)
    parser.add_argument('--phase',
                        dest='phase',
                        type=str,
                        choices=['train', 'test'],
                        required=True)
    parser.add_argument('--window-size',
                        dest='window_size',
                        type=int,
                        required=True)
    parser.add_argument('--window-offset',
                        dest='window_offset',
                        type=int,
                        required=True)
    args = parser.parse_args()

    tf_writer = tf.io.TFRecordWriter(os.path.join(args.output_dir, '{}_{}.tfrecord.gz'.format(
        args.phase, args.target)), tf.io.TFRecordCompressionType.GZIP)

    for root, dirs, files in os.walk(args.input_dir):
        frame_window = []
        if len(files) > 0:
            # logger.info('sort video frames')
            logger.info('encoding {}'.format(root))
            frame_id_list = [int(f.split('frame_')[-1].split('_')[0])
                             for f in files]
            frame_id_sorted = np.argsort(frame_id_list)

            for frame_id in range(0, len(frame_id_sorted) - args.window_size, args.window_offset):
                feat_dict = {}

                feat_dict['target'] = float_feature([TARGET_DICT[args.target]])

                for i in range(args.window_size):
                    image_bytes = open(
                        os.path.join(root, files[frame_id_sorted[frame_id + i]]), 'rb').read()
                    feat_dict['frame_{}'.format(i)] = bytes_feature([
                        image_bytes])

                tf_example = tf.train.Example(
                    features=tf.train.Features(feature=feat_dict))
                tf_writer.write(tf_example.SerializeToString())
    tf_writer.close()
