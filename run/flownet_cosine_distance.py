from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import glob
import os
import sys
from vgg16 import *
from recontruct_tfmath_withbatch import *


tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features,labels,mode):
  """Model function for CNN."""

  """Add code for stacking two input frames together to get 6-channel input layer.
  Assuming this will be named input_layer and fed into Conv layer #1"""

  # split the stacked input frame and the reconstructed frame
  frame1,frame2,frame3 = tf.split(features,3,3)
  imageStack = tf.concat([frame1,frame2],3)
  
  # Convolutional Layer #1
  # Computes 64 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  conv1 = tf.layers.conv2d(
      inputs=imageStack,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
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

  #Deconvolution Layer 1 - upsample by 2
  deconv5 = tf.layers.conv2d_transpose(
      inputs=conv6,
      filters=512,
      kernel_size=[2,2],
      strides=2,
      padding='valid',
      activation=tf.nn.relu
  )

  #Stack Deconvolved layer with pool5 layer (which is downsampled conv5_1 layer
  deconv5_1 = tf.concat([deconv5,conv5_1],3)

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
      activation=None
  )

  flow5_deconv2 = tf.layers.conv2d_transpose(
      inputs=flow5_deconv,
      filters=1,
      kernel_size=[2,2],
      strides=2,
      padding='valid',
      activation=None
  )

 #Deconvolutional layer #2 - upsample by two
  deconv4 = tf.layers.conv2d_transpose(
      inputs=deconv5_1,
      filters=256,
      kernel_size=[2,2],
      strides=2,
      padding='valid',
      activation=tf.nn.relu
  )
  
  deconv4_1 = tf.concat([deconv4,conv4_1,flow5_deconv2],3)

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
      activation=None
  )

  flow4_deconv2 = tf.layers.conv2d_transpose(
      inputs=flow4_deconv,
      filters=1,
      kernel_size=[2,2],
      strides=2,
      padding='valid',
      activation=None
  )

  #Deconvolutional Layer #3
  deconv3 = tf.layers.conv2d_transpose(
      inputs=deconv4_1,
      filters=128,
      kernel_size=[2,2],
      strides=2,
      padding='valid',
      activation=tf.nn.relu
  )

  deconv3_1 =  tf.concat([deconv3,conv3_1,flow4_deconv2],3)

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
      activation=None
  )

  flow3_deconv2 = tf.layers.conv2d_transpose(
      inputs=flow3_deconv,
      filters=1,
      kernel_size=[2,2],
      strides=2,
      padding='valid',
      activation=None
  )

  #Deconvolutional Layer #4
  deconv2 = tf.layers.conv2d_transpose(
      inputs=deconv3_1,
      filters=64,
      kernel_size=[2,2],
      strides=2,
      padding='valid',
      activation=tf.nn.relu
  )
  
  deconv2_1 =  tf.concat([deconv2,conv2,flow3_deconv2],3)

  predict_resize = tf.layers.conv2d_transpose(
    inputs=deconv2_1,
    filters=193,
    kernel_size=[2,2],
    strides=2,
    padding='valid',
    activation=None
  )
    
  flow_prediction = tf.layers.conv2d(
      inputs=predict_resize,
      filters=2,
      kernel_size=[5,5],
      padding='same',
      activation=tf.nn.relu
  )

#  flow_prediction = tf.concat([flow_prediction_a, tf.zeros([1,512,1024,1],tf.float32)],3)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=flow_prediction)


  # Calculate Loss (for both TRAIN and EVAL modes)
  #using cosine distance as a measure of directionality between predicited optical flow vectors and labels
  constants = tf.ones([tf.shape(flow_prediction)[1],tf.shape(flow_prediction)[2],tf.shape(flow_prediction)[3]])
  temp_tanh_pred = tf.tanh(flow_prediction)
  pred_scaled = tf.div(tf.add(temp_tanh_pred,constants),2)
  temp_tanh_labels = tf.tanh(labels)
  labels_scaled = tf.div(tf.add(temp_tanh_labels, constants), 2)


  loss = tf.losses.cosine_distance(
      labels = labels_scaled,
      predictions=pred_scaled,
      reduction=tf.losses.Reduction.MEAN,
      axis=3
  )


  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(
      learning_rate=0.001,
      beta1=0.9,
      beta2=0.999
    )
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=flow_prediction)}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
