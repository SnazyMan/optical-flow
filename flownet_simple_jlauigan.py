from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import glob
import os
import sys

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

  # Convolutional Layer #3
  # Computes 128 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 64]
  # Output Tensor Shape: [batch_size, 128, 128, 128]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 128, 128, 128]
  # Output Tensor Shape: [batch_size, 64, 64, 128]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #4
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 128]
  # Output Tensor Shape: [batch_size, 128, 128, 256]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  # Convolutional Layer #4.1
  # Computes 256 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 256]
  # Output Tensor Shape: [batch_size, 128, 128, 256]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #4
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 128, 128, 256]
  # Output Tensor Shape: [batch_size, 64, 64, 256]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #5
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 256]
  # Output Tensor Shape: [batch_size, 64, 64, 512]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  # Convolutional Layer #5.1
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 512]
  # Output Tensor Shape: [batch_size, 64, 64, 512]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #5
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 64, 64, 512]
  # Output Tensor Shape: [batch_size, 32, 32, 512]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #6
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 512]
  # Output Tensor Shape: [batch_size, 32, 32, 512]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)


  # Convolutional Layer #6.1
  # Computes 512 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 512]
  # Output Tensor Shape: [batch_size, 32, 32, 512]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #5
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32, 32, 512]
  # Output Tensor Shape: [batch_size, 16, 16, 512]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #7
  # Computes 1024 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 16, 512]
  # Output Tensor Shape: [batch_size, 16, 16, 512]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=1024,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

"""Need to add refinement layer from Flownet Paper"""

def main(unused_argv):

"""Import .flo file for labels - pass name of .flo file as argument"""
# WARNING: this will work on little-endian architectures (eg Intel x86) only!

        with open("/other-gt-flow/Grove2/flow10.flo", 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print 'Magic number incorrect. Invalid .flo file'
            else:
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                print 'Reading %d x %d flo file' % (w, h)
                data = np.fromfile(f, np.float32, count=2*w*h)
                # Reshape data into 3D array (columns, rows, bands)
                train_labels = np.resize(data, (h, w, 2))


"""Initialize training data and labels, save into tfrecord"""
train_data_path = "/other-data/Grove2/*.png"

addrs = glob.glob(train_data_path)
