import tensorflow as tf
from data_read import *
from flownet_simple import *

def main():
	# seting the path and dataset [to run on floydhub, mount at sintel]
	filename = ("/sintel")
	data = "albedo"

	# read the data
	input = get_data(filename,data)
	batched = input.batch(10);
	#iterators over the dataset
	features,labels = batched.make_one_shot_iterator().get_next()
	ofModel = tf.estimator.Estimator(cnn_model_fn(features,labels,tf.estimator.ModeKeys.TRAIN))        
	ofModel.train(input_fn=get_data(filename,data))

if __name__ == '__main__':
    main()
