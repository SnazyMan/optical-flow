
import tensorflow as tf
import numpy as np
import os

def read_flo(filename):
	"""Import .flo file for labels - pass name of .flo file as argument"""
	# WARNING: this will work on little-endian architectures (eg Intel x86) only!
	#with open(filename, 'rb') as f:
	#	magic = np.fromfile(f, np.float32, count=1)
	#	if 202021.25 != magic:
	#		print('Magic number incorrect. Invalid .flo file')
	#	else:
	#		w = np.fromfile(f, np.int32, count=1)[0]
	#		h = np.fromfile(f, np.int32, count=1)[0]
	#		# print('Readofing %d x %d flo file' % (h, w))
	#		data = np.fromfile(f, np.float32, count=2*w*h)
	#		# Reshape data into 3D array (columns, rows, bands
	#		train_labels = np.resize(data, (512, w, 2))
	#		train_labels = tf.convert_to_tensor(train_labels)
	train_labels = tf.read_file(filename)
	train_labels = tf.decode_raw(train_labels,tf.float32,True)
	# slice the first 3 bytes off of label. Byte 0 is magic number. Byte 1 is height. Byte 2 is width
	# slice semantics [start:stop:step] where we are zero based index 0,1,2,3,4 ...
	# remove 3 leading bytes [3:] - keep 3 to end       
	train_labels_s = train_labels[3:]
	# reshape to 512x1024x2
	train_labels_s_r = tf.reshape(train_labels_s,[436,1024,2])
	# Bilinear interpolate image to new size        
	train_labels_s_r_r = tf.image.resize_images(train_labels_s_r, [512,1024])
	#train_labels_s_r_r_s = tf.squeeze(train_labels_s_r_r,axis=0)        
	return train_labels_s_r_r

def parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string,channels=3)
  image_resized = tf.image.resize_images(image_decoded, [512, 1024])
  return image_resized

def set_value(matrix, x, y, val):
    w = int(matrix.get_shape()[0])
    h = int(matrix.get_shape()[1])
    val_diff = tf.reshape((tf.cast((val - matrix[x][y][0]),tf.float32)),[1])
    diff_matrix = tf.cast(tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[x, y, 0]], values=val_diff, dense_shape=[w, h,1])),tf.float32)
    matrix = tf.add(matrix,diff_matrix)
    return matrix

def recon_single_channel(split,flow):
	inter_split = tf.zeros([512,1024,1])
	for i in range(0,20):
		for j in range(0,20):
			temp_x = i+tf.div(flow[i,j,0],2)
			temp_y = j+tf.div(flow[i,j,1],2)
			temp_x = tf.cast(temp_x,tf.int64)
			temp_y = tf.cast(temp_y,tf.int64)
			inter_split=set_value(inter_split,temp_x,temp_y,split[i,j,0])
	return inter_split

def recon_single_frame(frame_temp,flow_temp):
	split0, split1, split2 = tf.split(frame_temp,3,2)
	inter_split0 = recon_single_channel(split0,flow_temp)
	inter_split1 = recon_single_channel(split1,flow_temp)
	inter_split2 = recon_single_channel(split2,flow_temp)
	# each inter_frame will be [1,512,1024,3]
	inter_frame = tf.concat([inter_split0,inter_split1,inter_split2], 2)
	inter_frame = tf.reshape(inter_frame,[1,512,1024,3])
	return inter_frame

def reconstuction(frames,flows,batch):
	'''
	frames : [batch, 512, 1024, 3]
	flows : [batch, 512, 1024, 2]
	return : inter_frames: [bacth, 512, 1024, 3]
	'''
	
	frame_temp  = tf.reshape(frames[0,:,:,:],[512,1024,3])
	flow_temp  = tf.reshape(flows[0,:,:,:],[512,1024,3])
	inter_frames = recon_single_frame(frame_temp,flow_temp)
	# frame will be of size [batch,512,1024,3]
	for i in range(1,batch):
		# each frame_temp will be of size [512,1024,3]
		frame_temp  = tf.reshape(frames[i,:,:,:],[512,1024,3])
		flow_temp  = tf.reshape(flows[i,:,:,:],[512,1024,3])
		inter_frames = tf.concat([inter_frames,recon_single_frame(frame_temp,flow_temp)], 0)
	return inter_frames

def main():
	# read in the file
	frame_path = ('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete/training/final/alley_1/frame_0001.png')
	flow_path = ('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete/training/flow/alley_1/frame_0001.flo')
	# if the reconstuction is put in out layers, the frame should be the first one in the pair
	# the flow should be the current prediction
	frame = parse_function(frame_path)
	frame = tf.concat([frame,frame],0)
	frames = tf.reshape(frame,[2,512,1024,3])
	flow = read_flo(flow_path)
	flow = tf.concat([flow,flow],0)
	flows = tf.reshape(flow,[2,512,1024,2])
	#init =tf.variables_initializer([inter_split0])
	#sess.run(init)
	# for i in range(0,4):
	# 	for j in range(0,4):
	# 		temp_x = i+tf.div(flow[i,j,0],2)
	# 		temp_y = j+tf.div(flow[i,j,1],2)
	# 		temp_x = tf.cast(temp_x,tf.int64)
	# 		temp_y = tf.cast(temp_y,tf.int64)
	# 		#set_value(inter_split0,temp_x,temp_y,split0[i,j,0])
	# 		w = int(inter_split0.get_shape()[0])
	# 		h = int(inter_split0.get_shape()[1])
	# 		val_diff = tf.reshape((tf.cast((split0[i,j,0] - inter_split0[temp_x][temp_x][0]),tf.float32)),[1])
	# 		diff_matrix = tf.cast(tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[temp_x, temp_y, 0]], values=val_diff, dense_shape=[w, h,1])),tf.float32)
	# 		inter_split0 = tf.add(inter_split0,diff_matrix)
	batchsize = 2
	inter_frames = reconstuction(frames,flows,batchsize)
	sess = tf.Session()
	print(sess.run(tf.shape(inter_frames)))
	#print(sess.run(tf.shape(split0)))
	



if __name__ == '__main__':
	main()
