import tensorflow as tf
from data_read import *
from flownet_simple import *

def main():
	# seting the path and set [to run on floydhub, mount at sintel]
	filename = ("/sintel")
	#filename= "/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete"
	data_train = "albedo"
	data_test ="clean"


	#dataset = get_data(filename,data)
	#iterator = dataset.make_one_shot_iterator()
	#features,labels = iterator.get_next()
	#with tf.Session() as sess: sess.run(print(type(one_element[1])))

	ofModel = tf.estimator.Estimator(model_fn=cnn_model_fn)

	ofModel.train(input_fn=lambda:get_data(filename,data_train))

	eval = ofModel.evaluate(input_fn=lambda:get_data(filename,data_train))

	print(eval)

	predictions = ofModel.predict(
		x = None,
		input_fn=lambda:get_data_test(filename,data_clean),
		batch_size=None,
		outputs=None,
		as_iterable=True,
		iterate_batches = false
		)

	for e in predictions:
		print(type(e))

if __name__ == '__main__':
	main()


    
