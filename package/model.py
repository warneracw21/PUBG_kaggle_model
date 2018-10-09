from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Import System Libraries
import sys
import os
from functools import partial

# Import Modules
import tensorflow as tf

# Import Libraries and Configs
import config

def model_fn(features, labels, mode, params):

    # Define Modes
    train = mode == tf.estimator.ModeKeys.TRAIN
    test_ = mode == tf.estimator.ModeKeys.PREDICT
    eval_ = mode == tf.estimator.ModeKeys.EVAL


    # Define Model
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.layers.batch_normalization(net, training=train, name='normalize')

    with tf.variable_scope('DNN', reuse=tf.AUTO_REUSE):
        for layer_num, layer_size in enumerate(params['layer_sizes']):
            with tf.variable_scope('dense_%d' % layer_num):
                w = tf.get_variable(
                            name='weight_%d' % layer_num,
                            dtype=tf.float32,
                            shape=[net.shape[-1], layer_size],
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(
                                                            params['weight_decay']))
                b = tf.get_variable(
                            name='bias_%d' % layer_num,
                            dtype=tf.float32,
                            shape=[layer_size],
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(
                                                            params['weight_decay']))
                net = tf.tensordot(net, w, axes=1) + b

                # Last Layer Output should be probabilistic
                if (layer_num + 1) == len(params['layer_sizes']):
                    net = tf.sigmoid(net)
                else:
                    net = tf.sigmoid(net)

                    # Add a dropout for training every layer save last
                    if train:
                        net = tf.nn.dropout(net, keep_prob=0.5)
        
        # Reduce the last dimension for evaluation
        net = tf.squeeze(net, name='squeeze')


    # Define Metrics
    loss = tf.losses.mean_squared_error(
                            labels=labels,
                            predictions=net,
                            scope='MSE')

    with tf.variable_scope('accuracy'):
        accuracy = tf.metrics.accuracy(
                                labels=labels,
                                predictions=net,
                                name='acc_op')


    # Return Estimator Specs
    if test_:
        predictions = {
            'value': net
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
    if train:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if eval_ == tf.estimator.ModeKeys.EVAL:
        tf.summary.scalar('accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops={'accuracy': accuracy})



