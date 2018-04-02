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
