from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import glob
import os
import sys
from data_read import *

tf.logging.set_verbosity(tf.logging.INFO)

def load_image(addr):
    # read an image and resize to (256, 256)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  #Tensor object to keep track of pooling argmax
  unpool_tensor = Tensor();
  
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # Middlebury images are 256x256 pixels, three color channel
  input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

  # Convolutional Layer #1
  # Computes 6 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 256, 256, 3]
  # Output Tensor Shape: [batch_size, 256, 256, 6]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=6,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 256, 256, 6]
  # Output Tensor Shape: [batch_size, 128, 128, 6]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  #pool1,argmax = tf.nn.max_pool_with_argmax(input = conv1, ksize = [2,2], strides = 2)
  #unpool_tensor = tf.concat([unpool_tensor,argmax],0)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 6]
  # Output Tensor Shape: [batch_size, 128, 128, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 128, 128, 64]
  # Output Tensor Shape: [batch_size, 64, 64, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  #pool2,argmax = tf.nn.max_pool_with_argmax(input = conv2, ksize = [2,2], strides = 2)
  #unpool_tensor = tf.concat([unpool_tensor,argmax],0)
  
  # Convolutional Layer #3
  # Computes 128 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 64]
  # Output Tensor Shape: [batch_size, 128, 128, 128]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 128, 128, 128]
  # Output Tensor Shape: [batch_size, 64, 64, 128]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Convolutional Layer #4
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 128]
  # Output Tensor Shape: [batch_size, 128, 128, 256]
  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  # Convolutional Layer #4.1
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 256]
  # Output Tensor Shape: [batch_size, 128, 128, 256]
  conv4_1 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #4
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 128, 128, 256]
  # Output Tensor Shape: [batch_size, 64, 64, 256]
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  # Convolutional Layer #5
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 256]
  # Output Tensor Shape: [batch_size, 64, 64, 512]
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
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

  # Convolutional Layer #6
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 512]
  # Output Tensor Shape: [batch_size, 32, 32, 512]
  conv6 = tf.layers.conv2d(
      inputs=pool5,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  # Convolutional Layer #6.1
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 512]
  # Output Tensor Shape: [batch_size, 32, 32, 512]
  conv6_1 = tf.layers.conv2d(
      inputs=conv6,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #6
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32, 32, 512]
  # Output Tensor Shape: [batch_size, 16, 16, 512]
  pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

  # Convolutional Layer #7
  # Computes 1024 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 16, 512]
  # Output Tensor Shape: [batch_size, 16, 16, 512]
  conv7 = tf.layers.conv2d(
      inputs=pool6,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  #Deconvolution Layer 1
  deconv7 = tf.layers.conv2d_transpose(
      inputs=conv7,
      filters=512,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  deconv7_1 = tf.concat([deconv7,conv6_1],0)

  flow6 = tf.layers.conv2d(
      inputs=deconv7_1,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  flow6_deconv = tf.layers.conv2d_transpose(
      inputs=flow6,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation='None'
  )

  flow6_deconv2 = tf.layers.conv2d_transpose(
      inputs=flow6_deconv,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation='None'
  )

  deconv6 = tf.layers.conv2d_transpose(
      inputs=deconv7_1,
      filters=256,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  deconv6_1 = tf.concat([deconv6,conv5_1,flow6_deconv2],0)

  flow5 = tf.layers.conv2d(
      inputs=deconv6_1,
      filters=1,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  flow5_deconv = tf.layers.conv2d_transpose(
      inputs=flow5,
      filters=1,
      kernel_size=[5, 5],
      padding='valid',
      activation='None'
  )

  flow5_deconv2 = tf.layers.conv2d_transpose(
      inputs=flow5_deconv,
      filters=1,
      kernel_size=[5, 5],
      padding='valid',
      activation='None'
  )

  deconv5 = tf.layers.conv2d(
      inputs=deconv6_1,
      filters=128,
      kernel_size=[5,5],
      padding='valid',
      activation=tf.nn.relu
  )

  deconv5_1 =  tf.concat([deconv5,conv4_1,flow5_deconv2],0)
