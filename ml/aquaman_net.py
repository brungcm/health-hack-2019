#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Wed 13 Sep 2018 18:00:00

from __future__ import division

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def _conv2d(_input, n_filters, kernel_size, reg_val, name=None):
    return tf.layers.conv2d(_input, filters=n_filters,
                            kernel_size=kernel_size,
                            strides=(1, 1),
                            activation=tf.nn.elu,
                            padding='valid',
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                reg_val),
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            bias_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            bias_regularizer=tf.contrib.layers.l2_regularizer(
                                reg_val),
                            name=name)


def _conv(_input, n_filters, kernel_size, reg_val, name=None):
    return tf.layers.conv3d(_input, filters=n_filters,
                            kernel_size=kernel_size,
                            strides=(1, 1, 1),
                            activation=tf.nn.elu,
                            padding='valid',
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                reg_val),
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            bias_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            bias_regularizer=tf.contrib.layers.l2_regularizer(
                                reg_val),
                            name=name)


def _dense(_input, n_outputs, reg_val, name=None):
    return tf.layers.dense(_input, units=n_outputs,
                           activation=tf.nn.elu,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(
                               reg_val),
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                           bias_initializer=tf.contrib.layers.variance_scaling_initializer(),
                           bias_regularizer=tf.contrib.layers.l2_regularizer(
                               reg_val),
                           name=name)


def AquamanNet(input_tensor, is_training=True, n_classes=2):
    reg_val = 5e-4
    dropout = 0.4

    with tf.variable_scope("AquamanNet"):

        net = _conv(input_tensor, 32, (3, 3, 3), reg_val,
                    name='netconv_0_0')
        print(net)
        net = _conv(input_tensor, 32, (3, 3, 3), reg_val,
                    name='netconv_0_1')
        print(net)
        net = tf.layers.max_pooling3d(
            net, (2, 2, 2), (2, 2, 2), padding='same', name='netMaxPool_0')
        print(net)

        net = _conv(net, 32, (3, 3, 2), reg_val,
                    name='netconv_1_1')
        net = _conv(net, 32, (3, 3, 2), reg_val,
                    name='netconv_1_2')
        net = tf.layers.max_pooling3d(
            net, (2, 2, 2), (2, 2, 2), padding='same', name='netMaxPool_1')

        net = _conv(net, 32, (3, 3, 2), reg_val,
                    name='netconv_2_1')
        net = _conv(net, 32, (3, 3, 2), reg_val,
                    name='netconv_2_2')
        net = tf.layers.max_pooling3d(
            net, (2, 2, 2), (2, 2, 3), padding='same', name='netMaxPool_2')

        print(net)
        print('\n\n\n')

        net = tf.squeeze(net, axis=3, name='net_squeeze_0')

        net = _conv2d(net, 32, (3, 3), reg_val,
                      name='netconv_3_1')
        net = _conv2d(net, 32, (3, 3), reg_val,
                      name='netconv_3_2')
        net = tf.layers.max_pooling2d(
            net, (2, 2), (2, 2), padding='same', name='netMaxPool_3')
        print(net)
        print('\n\n\n')

        net = _conv2d(net, 32, (3, 3), reg_val,
                      name='netconv_4_1')
        net = _conv2d(net, 32, (3, 3), reg_val,
                      name='netconv_4_2')
        net = tf.layers.max_pooling2d(
            net, (2, 2), (2, 2), padding='same', name='netMaxPool_4')
        print(net)
        print('\n\n\n')

        net = tf.layers.flatten(net)
        print("Flatten")
        print(net)

        net = _dense(net, 256, reg_val, 'net_dense_1')
        net = tf.layers.dropout(net, rate=dropout, name='net_dropout_1')

        net = _dense(net, 64, reg_val, 'net_dense_2')
        net = tf.layers.dropout(net, rate=dropout, name='net_dropout_2')

        net = _dense(net, n_classes, reg_val, 'net_dense_3')

        # import sys
        # sys.exit(0)

        return net
