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

def input_fn(path):

	with tf.variable_scope('input_fn'):
	
		# Read in the CSV one line at a time
		dataset = tf.data.TextLineDataset(path).skip(1)

		# Map the data into the specific features
		dataset = dataset.map(_parse_line)
		dataset = dataset.repeat().shuffle(buffer_size=512)

		# Batch and return iterator
		dataset = dataset.batch(config.BATCH_SIZE)
		iterator = dataset.make_one_shot_iterator()
		
		return iterator.get_next()

def _parse_line(line):

	# Decode the line into fields
	fields = tf.decode_csv(line, config.FIELD_DEFAULTS, select_cols=[i for i in range(3,27)])

	# Pack the Results into a dictionary
	features = dict(zip(config.COLUMN_NAMES, fields))

	# Pop out of the label
	label = features.pop('winPlacePerc')

	return features, label


