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

	ofModel.train(input_fn=lambda:get_data(filename,data,interpath,True))

	eval_train = ofModel.evaluate(input_fn=lambda:get_data(filename,data,interpath,True))
	eval_test = ofModel.evaluate(input_fn=lambda:get_data(filename,data,interpath,False))


	print(eval_train)
	print(eval_test)

	#sess = tf.Session()
	#saver = tf.train.import_meta_graph('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/DFG4eQPeJPbVovbKYr5XzX/model.ckpt-880.meta')
	#saver.restore(sess, tf.train.latest_checkpoint('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/DFG4eQPeJPbVovbKYr5XzX'))

	predictions = ofModel.predict(
		input_fn=lambda:get_data(filename,'final',interpath,False),
		predict_keys=None,
		hooks=None,
		checkpoint_path=None,
		)
	print(type(predictions))
	print(type(next(predictions)))
	print(next(predictions))
	

if __name__ == '__main__':
	main()


    
