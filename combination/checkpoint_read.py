import tensorflow as tf
import numpy as np 

ck = tf.train.latest_checkpoint('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/DFG4eQPeJPbVovbKYr5XzX')
reader = tf.train.NewCheckpointReader(ck)
all_variables = reader.get_variable_to_shape_map()
for key in all_variables:
	print(key)
w1 = reader.get_tensor("conv2d_10/kernel/Adam")
print(w1.shape)
print(w1)


#sess = tf.Session()
#saver = tf.train.import_meta_graph('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/DFG4eQPeJPbVovbKYr5XzX/model.ckpt-880.meta')
#saver.restore(sess, tf.train.latest_checkpoint('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/DFG4eQPeJPbVovbKYr5XzX'))
# predictions = ofModel.predict(
# 	input_fn=lambda:get_data(filename,'final',interpath,False),
# 	predict_keys=None,
# 	hooks=None,
# 	checkpoint_path=None,
# 	)
# print(type(predictions))
# print(type(next(predictions)))
# print(next(predictions))
