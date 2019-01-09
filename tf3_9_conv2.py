#coding:utf-8
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# the format of value: [NHWC]
value = tf.reshape(tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]), [1, 3, 3, 1]) 
# the format of filter: [height, width, output_channels, input_channels]
filter = tf.reshape(tf.constant([[1., 0.], [0., 1.]]), [2, 2, 1, 1])
# the format of output_shape: [NHWC]
output_shape = [1, 5, 5, 1]
# the format of strides: [1, stride, stride, 1]
strides = [1, 2, 2, 1]
padding = 'SAME'
# define the transpose conv op
transpose_conv = tf.nn.conv2d_transpose(value=value, filter=filter, output_shape=output_shape, strides=strides, padding=padding)
sess = tf.Session()
sess.run(transpose_conv)