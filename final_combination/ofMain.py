import tensorflow as tf
from data_read import *
from flownet_cosine_distance_absolute_difference_combined import *

def main():
	# seting the path and set [to run on floydhub, mount at sintel]
	#filename = ("/home/snazyman/machine_learning/optical-flow/data/sintel")
	filename = ("/sintel")        
	#interpath = ("/home/snazyman/machine_learning/optical-flow/data/inter_flow/inter_flow")
	interpath = ("/inter")               
	data = "final"
	#data_test = "clean"

	#dataset = get_data(filename,data_train,interpath)
	#iterator = dataset.make_one_shot_iterator()
	#features,labels = iterator.get_next()
	#with tf.Session() as sess: sess.run(print(type(one_element[1])))
	#cnn_model_fn(features,labels,None)        

	ofModel = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='/output')

	ofModel.train(input_fn=lambda:get_data(filename,data,True,50,1))

	eval_train = ofModel.evaluate(input_fn=lambda:get_data(filename,data,True,1,1))
	eval_test = ofModel.evaluate(input_fn=lambda:get_data(filename,data,False,1,1))

	##################### feed in one experiments
	#ofModel.train(input_fn=lambda:feed_one(filename))


	print(eval_train)
	print(eval_test)

	predictions = ofModel.predict(
	input_fn=lambda:feed_one(filename),
	predict_keys=None,
	hooks=None,
	checkpoint_path=None,
	)
	test_flow = next(predictions)
	flo_write(test_flow,"/output/alpha0_all.flo")

	# predictions = ofModel.predict(
	# input_fn=lambda:get_data(filename,data,True),
	# predict_keys=None,
	# hooks=None,
	# checkpoint_path=None,
	# )
	# test_flow = next(predictions)
	# print(type(predictions))
	# print(type(test_flow))
	# print(test_flow.shape)
	# print(test_flow)


if __name__ == '__main__':
	main()


    
