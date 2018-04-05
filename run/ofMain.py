import tensorflow as tf
from data_read import *
from flownet_simple import *

def main():
	# seting the path and dataset [to run on floydhub, mount at sintel]
	filename = ("/home/snazyman/machine_learning/optical-flow/data/sintel")
	data = "albedo"

        # create estimator object with flownet_simple network arch
	ofModel = tf.estimator.Estimator(model_fn=cnn_model_fn)

        # train the network
	ofModel.train(input_fn=lambda:get_data(filename,data))

if __name__ == '__main__':
    main()
