import tensorflow as tf
from data_read import *
from flownet_cosine_distance_absolute_difference_combined import *

def main():
	# seting the path and set [to run on floydhub, mount at sintel]
	#filename = ("/home/snazyman/machine_learning/optical-flow/data/sintel")
	filename = ("/sintel")        
	#interpath = ("/home/snazyman/machine_learning/optical-flow/data/inter_flow/inter_flow")
	interpath = ("/inter")               
	data_train = "final"
	data_test = "clean"

	#dataset = get_data(filename,data_train,interpath)
	#iterator = dataset.make_one_shot_iterator()
	#features,labels = iterator.get_next()
	#with tf.Session() as sess: sess.run(print(type(one_element[1])))
	#cnn_model_fn(features,labels,None)        

	ofModel = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='/output')

	ofModel.train(input_fn=lambda:get_data(filename,data_train,interpath))

	eval = ofModel.evaluate(input_fn=lambda:get_data(filename,data_train,interpath))

	print(eval)

	#predictions = ofModel.predict(
	#	x = None,
	#	input_fn=lambda:get_data_test(filename,data_clean),
	#	batch_size=None,
	#	outputs=None,
	#	as_iterable=True,
	#	iterate_batches = false
	#	)

	#for e in predictions:
	#	print(type(e))

if __name__ == '__main__':
	main()


    
