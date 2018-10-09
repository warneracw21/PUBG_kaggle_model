from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Import System Libraries
import sys
import os

# Import Modules
import tensorflow as tf

# Import Libraries and Configs
import config

def linear_regression(net, is_training=True, weight_decay=2.5e-5):

    net = tf.layers.batch_normalization(net, training=is_training)

    with tf.variable_scope('DNN', reuse=tf.AUTO_REUSE):
        layer_sizes = [512, 256, 128, 64, 1]
        for layer_num, layer_size in enumerate(layer_sizes):
            with tf.variable_scope('dense_%d' % layer_num):
                w = tf.get_variable(
                            name='weight_%d' % layer_num,
                            dtype=tf.float32,
                            shape=[net.shape[-1], layer_size],
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            trainable=is_training)
                b = tf.get_variable(
                            name='bias_%d' % layer_num,
                            dtype=tf.float32,
                            shape=[layer_size],
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            trainable=is_training)
                net = tf.tensordot(net, w, axes=1) + b
                net = tf.sigmoid(net)

    return tf.squeeze(net)

