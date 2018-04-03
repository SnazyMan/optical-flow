from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from data_read import *
import glob
import os
import sys

tf.logging.set_verbosity(tf.logging.INFO)



def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  """Add code for stacking two input frames together to get 6-channel input layer.
  Assuming this will be named input_layer and fed into Conv layer #1"""

  # Convolutional Layer #1
  # Computes 64 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


  # Convolutional Layer #2
  # Computes 128 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


  # Convolutional Layer #3
  # Computes 256 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3.1
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  conv3_1 = tf.layers.conv2d(
      inputs=conv3,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  pool3 = tf.layers.max_pooling2d(inputs=conv3_1, pool_size=[2, 2], strides=2)


  # Convolutional Layer #4
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #4.1
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  conv4_1 = tf.layers.conv2d(
      inputs=conv4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #4
  # Second max pooling layer with a 2x2 filter and stride of 2
  pool4 = tf.layers.max_pooling2d(inputs=conv4_1, pool_size=[2, 2], strides=2)



  # Convolutional Layer #5
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  conv5 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  # Convolutional Layer #5.1
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 512]
  # Output Tensor Shape: [batch_size, 64, 64, 512]
  conv5_1 = tf.layers.conv2d(
      inputs=conv5,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #5
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 64, 64, 512]
  # Output Tensor Shape: [batch_size, 32, 32, 512]
  pool5 = tf.layers.max_pooling2d(inputs=conv5_1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #6
  # Computes 1024 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 512]
  # Output Tensor Shape: [batch_size, 32, 32, 512]
  conv6 = tf.layers.conv2d(
      inputs=pool5,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  """Begin Refinement Layer
  For the tf.layers.conv2d_transpose - still not too sure on whether filter numbers are for output or input
  but will be easy to check when running"""

  #Deconvolution Layer 1
  deconv5 = tf.layers.conv2d_transpose(
      inputs=conv6,
      filters=512,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  #Stack Deconvolved layer with pool5 layer (which is downsampled conv5_1 layer
  deconv5_1 = tf.concat([deconv5,pool5],0)

  #Calculate intermediate flow frame
  flow5 = tf.layers.conv2d(
      inputs=deconv5_1,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  #Begin process of reshaping frame to correct size through deconvolution w/ no activation function
  #This is needed due to arbitrary input shape of input frames - but effectively just upsampling
  flow5_deconv = tf.layers.conv2d_transpose(
      inputs=flow5,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation='None'
  )

  flow5_deconv2 = tf.layers.conv2d_transpose(
      inputs=flow5_deconv,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation='None'
  )

 #Deconvolutional layer #2
  deconv4 = tf.layers.conv2d_transpose(
      inputs=deconv5_1,
      filters=256,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  deconv4_1 = tf.concat([deconv4,pool4,flow6_deconv2],0)

  flow4 = tf.layers.conv2d(
      inputs=deconv4_1,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  flow4_deconv = tf.layers.conv2d_transpose(
      inputs=flow4,
      filters=1,
      kernel_size=[5, 5],
      padding='valid',
      activation='None'
  )

  flow4_deconv2 = tf.layers.conv2d_transpose(
      inputs=flow4_deconv,
      filters=1,
      kernel_size=[5, 5],
      padding='valid',
      activation='None'
  )

  #Deconvolutional Layer #3
  deconv3 = tf.layers.conv2d(
      inputs=deconv4_1,
      filters=128,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  deconv3_1 =  tf.concat([deconv3,pool3,flow4_deconv2],0)

   flow3 = tf.layers.conv2d(
      inputs=deconv3_1,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
   )

  flow3_deconv = tf.layers.conv2d_transpose(
      inputs=flow3,
      filters=1,
      kernel_size=[5, 5],
      padding='valid',
      activation='None'
  )

  flow3_deconv2 = tf.layers.conv2d_transpose(
      inputs=flow3_deconv,
      filters=1,
      kernel_size=[5, 5],
      padding='valid',
      activation='None'
  )

  #Deconvolutional Layer #4
  deconv2 = tf.layers.conv2d(
      inputs=deconv3_1,
      filters=64,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  deconv2_1 =  tf.concat([deconv2,pool2,flow3_deconv2],0)

   flow_prediction = tf.layers.conv2d(
      inputs=deconv2_1,
      filters=2,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )


  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=flow_prediction)

  """loss is euclidean distance, output is same shape as labels - pixel based loss"""
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.absolute_difference(
      labels=labels,
      predictions=flow_prediction,
      weights=1.0,
      reduction='NONE'
  )

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)





def main():
 	# seting the path and dataset
 	filename = ("/sintel")
 	data = "albedo"

 	# read the data
 	input = get_data(filename,data)

        #iterators over the dataset
 	iterator = input.make_one_shot_iterator()
 	one_element = iterator.get_next()
 	with tf.Session() as sess:
            for i in range(5):
                print("hi")
                print(type(one_element[0]))
                print(type(one_element[1]))
                print(type(one_element[2]))

if __name__ == '__main__':
    main()
