from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Import Libraries
import sys
import os
from functools import partial 

# Import Modules
import tensorflow as tf

# Import Libraries and Configs
import config
import data_provider
import model

# Set Parameters
tf.logging.set_verbosity(tf.logging.INFO)

train_columns = [col_name for col_name in config.COLUMN_NAMES]
train_columns.remove('winPlacePerc')

feature_columns = [tf.feature_column.numeric_column(col_name) 
						for col_name in train_columns]
layer_sizes = [1024, 512, 256, 128, 64, 1]
weight_decay = 2.5e-5

regression_estimator = tf.estimator.Estimator(
								model_fn=model.model_fn,
								model_dir=os.path.join(config.PATH_TO_PROJECT, 'models/'),
								params={
									'feature_columns': feature_columns,
									'layer_sizes': layer_sizes,
									'weight_decay': weight_decay
								})

train_spec = tf.estimator.TrainSpec(
					input_fn=partial(data_provider.input_fn, path=config.TRAIN_PATH),
					max_steps=int(1e6))
eval_spec = tf.estimator.EvalSpec(
					input_fn=partial(data_provider.input_fn, path=config.EVAL_PATH),
					start_delay_secs=600,
					throttle_secs=600)

tf.estimator.train_and_evaluate(
					estimator=regression_estimator,
					train_spec=train_spec,
					eval_spec=eval_spec)