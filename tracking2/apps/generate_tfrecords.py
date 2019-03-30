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

import mlflow
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors
from sklearn.model_selection import KFold

from utils import utils


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def leave_one_session_out(dataset,
                          output_dir,
                          pipeline_args,
                          window_size_sec,
                          window_offset_sec,
                          frames_per_window,
                          frames_offset,
                          target_frequency_list,
                          signal_mean=0,
                          signal_std=1,
                          cutoff_freq=40,
                          apply_car=False):
    """ Encode tf-records in an K-fold strategy per subject 
        given a list of frequency of interest

        OUTPUT:
            <output-path>/<protocol>/<datetime>/<subject>/<fold_k>/<train/test>
    """

    logger.info("window_size_sec {}".format(window_size_sec))
    logger.info("window_offset_sec {}".format(window_offset_sec))
    logger.info("frames_per_window {}".format(frames_per_window))
    logger.info("frames_offset {}".format(frames_offset))
    logger.info("target_frequency_list {}".format(target_frequency_list))
    logger.info("signal_mean {}".format(signal_mean))
    logger.info("signal_std {}".format(signal_std))

    datetime_now_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # mlflow.log_param("timestamp", datetime_now_str)

    logger.info('Saving target dictionary')
    full_path_output_dir = os.path.join(
        output_dir, 'leave_one_session_out', datetime_now_str)    

    os.makedirs(full_path_output_dir)
    target_dict_i2f = {k: v for k, v in enumerate(target_frequency_list)}
    target_dict_f2i = {v: k for k, v in target_dict_i2f.iteritems()}
    target_json = json.dumps(target_dict_i2f)
    with open(os.path.join(full_path_output_dir, 'target_dict.json'), 'w') as f:
        f.write(target_json)

    logger.info('Target dictionary: {}'.format(target_dict_i2f))
    logger.info('Target dictionary created')

    leave_one_session_out.n_frames_per_trial = None

    for subject in dataset:
        logger.info(subject)
        fs_idx_list = []
        for f in target_frequency_list:
            f_idx_list = []
            for trial_id, trial in enumerate(dataset[subject]):
                if trial[0] == f:
                    f_idx_list.append(trial_id)
            fs_idx_list.append(f_idx_list)

        trials_folds = zip(*fs_idx_list)
        kf = KFold(n_splits=len(trials_folds))

        for fold_id, (train_index, test_index) in enumerate(kf.split(trials_folds)):
            logger.info('FOLD: {}'.format(fold_id))

            full_path_output_dir_subject_trial = os.path.join(
                full_path_output_dir, subject, str(fold_id))

            os.makedirs(full_path_output_dir_subject_trial)

            train_filename = os.path.join(
                full_path_output_dir_subject_trial, 'train.tfrecord.gz')
            test_filename = os.path.join(
                full_path_output_dir_subject_trial, 'test.tfrecord.gz')

            train_writer = tf.io.TFRecordWriter(
                train_filename, tf.io.TFRecordCompressionType.GZIP)
            test_writer = tf.io.TFRecordWriter(
                test_filename, tf.io.TFRecordCompressionType.GZIP)

            def _dump_fold(writer, fold_index):

                for fold in fold_index:
                    for signal_id in trials_folds[fold]:

                        time_signal_all_electrodes = dataset[subject][signal_id][1]
                        if apply_car:
                            time_signal_all_electrodes = utils.apply_car(
                                time_signal_all_electrodes)

                        time_signal_target = target_dict_f2i[dataset[subject]
                                                             [signal_id][0]]

                        power_stack_dict = {}
                        for electrode_id in range(time_signal_all_electrodes.shape[1]):
                            power_stack, _, _, freq_list = utils.get_stacked_spectrum(signal=time_signal_all_electrodes[:, electrode_id],
                                                                                      window_size=window_size_sec,
                                                                                      window_offset=window_offset_sec,
                                                                                      sampling_rate=utils.SSVEP_SR)
                            # perform z-norm
                            power_stack = (
                                power_stack - signal_mean) / signal_std

                            cutoff_freq_idx = np.where(
                                freq_list >= cutoff_freq)[0][0]

                            stacked_frames = []
                            for ws in range(0, power_stack.shape[1] - frames_per_window, frames_offset):
                                stacked_frames.append(
                                    power_stack[:cutoff_freq_idx, ws:ws + frames_per_window])

                            power_stack_dict['electrode_' +
                                             str(electrode_id)] = stacked_frames

                            if leave_one_session_out.n_frames_per_trial is None:
                                leave_one_session_out.n_frames_per_trial = len(stacked_frames)

                                with open(os.path.join(full_path_output_dir, 'frame_shape.txt'), 'w') as f:
                                    f.write(
                                        str(stacked_frames[0].shape[0]) + '\n')
                                    f.write(str(stacked_frames[0].shape[1]))

                        for i in range(leave_one_session_out.n_frames_per_trial):
                            feat_dict = {}
                            feat_dict['target'] = float_feature(
                                [time_signal_target])
                            feat_dict['target_frequency'] = bytes_feature(
                                [utils.SSVEP_LABEL2STR[dataset[subject][signal_id][0]]])
                            feat_dict['subject'] = bytes_feature(
                                [str(subject)])
                            feat_dict['trial'] = int_feature([signal_id])
                            feat_dict['frame_id'] = int_feature([i])

                            for electrode_id in power_stack_dict:
                                squeezed_signal = np.squeeze(
                                    power_stack_dict[electrode_id][i].reshape(1, -1))

                                feat_dict[electrode_id] = float_feature(
                                    squeezed_signal)

                            tf_example = tf.train.Example(
                                features=tf.train.Features(feature=feat_dict))
                            writer.write(tf_example.SerializeToString())

            logger.info('Train')
            _dump_fold(train_writer, train_index)

            logger.info('Test')
            _dump_fold(test_writer, test_index)

            train_writer.close()
            test_writer.close()
    return full_path_output_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir',
                        dest='input_data',
                        type=str,
                        required=True)
    parser.add_argument('--output-dir',
                        dest='output_dir',
                        type=str,
                        required=True)
    parser.add_argument('--window-size',
                        dest='window_size',
                        type=int,
                        required=True)
    args, pipeline_args = parser.parse_known_args()

