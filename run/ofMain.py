import tensorflow as tf
from data_read import *
from flownet_simple import *

def main():
	# seting the path and set [to run on floydhub, mount at sintel]
	filename = ("/sintel")
	data = "albedo"

	#dataset = get_data(filename,data)
	#iterator = dataset.make_one_shot_iterator()
	#features,labels = iterator.get_next()
	#with tf.Session() as sess: sess.run(print(type(one_element[1])))

	ofModel = tf.estimator.Estimator(model_fn=cnn_model_fn)

	ofModel.train(input_fn=lambda:get_data(filename,data))

	eval = ofModel.evaluate(input_fn=lambda:get_data(filename,data))

	print(eval)

if __name__ == '__main__':
	main()


    
