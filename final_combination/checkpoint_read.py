import tensorflow as tf
import numpy as np 
import tensorflow as tf
from data_read import *
from flownet_cosine_distance_absolute_difference_combined import *
from flo_write import * 
data = 'final'
ofModel = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='/Users/renzhihuang/Desktop/final_cnn/#88 batch4/oBskF5GUzB8jYRMC5hLUvd')
filename = ("/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete")        
#eval_train = ofModel.evaluate(input_fn=lambda:get_data(filename,data,True))
#eval_test = ofModel.evaluate(input_fn=lambda:get_data(filename,data,False))
#ck = tf.train.latest_checkpoint('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/adam_0.1_alpha0.5')
#### check weights and bias
# reader = tf.train.NewCheckpointReader(ck)
# all_variables = reader.get_variable_to_shape_map()
# for key in all_variables:
# 	print(key)
# w1 = reader.get_tensor("conv2d_3/kernel/Adam")
# print(w1.shape)
# print(w1)


####check predictions
sess = tf.Session()
#saver = tf.train.import_meta_graph('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/adam_0.1_alpha0.5/model.ckpt-1.meta')
#saver.restore(sess, '/Users/renzhihuang/Desktop/CIS520/project/tensorflow/adam_0.1_alpha0.5/model.ckpt-1.meta')
predictions = ofModel.predict(
	input_fn=lambda:feed_one(filename),
	predict_keys=None,
	hooks=None,
	checkpoint_path=None,
	)
test_flow = next(predictions)
print(type(predictions))
print(type(test_flow))
print(test_flow.shape)
print(test_flow)
flo_write(test_flow,"/Users/renzhihuang/Desktop/final_cnn/#88 batch4/oBskF5GUzB8jYRMC5hLUvd/alpha0_all.flo")
