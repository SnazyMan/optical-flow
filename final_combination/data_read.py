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
	#train_labels_s_r_r = tf.image.per_image_standardization(train_labels_s_r_r)
	#train_labels_s_r_r_s = tf.squeeze(train_labels_s_r_r,axis=0)        
	return train_labels_s_r_r

def read_flo2(filename):
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
	train_labels_s_r = tf.reshape(train_labels_s,[512,1024,2])
	# reshape to 512x1024x2
	# Bilinear interpolate image to new size        
	train_labels_s_r = tf.image.per_image_standardization(train_labels_s_r)
	#train_labels_s_r_r_s = tf.squeeze(train_labels_s_r_r,axis=0)        
	return train_labels_s_r

def parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string,channels=3)
  image_resized = tf.image.resize_images(image_decoded, [512, 1024])
  image_resized = tf.image.per_image_standardization(image_resized)
  return image_resized

def get_data(filename,data_name,train,repeat,batch):
	''' filename: the path of MPI-Sintel-complete
	    data_name: the dataset we use (albedo, clean or ...)'''

	# get the sub directory for this dataset
	path = filename+"/training/"+data_name
	subdir = next(os.walk(path))[1]
	subdir.sort()

	
        
	# read only 4 sub directories
	subdir = [subdir[x] for x in range(0,1)]

	# get the list of file names
	# filename1: list of frame1 tensor
	# filename2: list of frame2 tensor
	# ground_truth_flow: list of flow tensor
	filenames1 = []
	filenames2 = []
	ground_truth_flow = [];
	for sub in subdir:
		number = len(next(os.walk(filename+"/training/"+data_name+"/"+sub))[2])
		testset = 30
		if train == True:
			low = 1
			up = number - testset-1
		else:
			low = number - testset
			up = number
		for i in range(low,up):
			if i < 10:
				filenames1.append(parse_function(filename+"/training/%s/%s/frame_000%d.png" % (data_name,sub,i)))
			else:
				filenames1.append(parse_function(filename+"/training/%s/%s/frame_00%d.png" % (data_name,sub,i)))
		for i in range(low+1,up+1):
			if i < 10:
				filenames2.append(parse_function(filename+"/training/%s/%s/frame_000%d.png" % (data_name,sub,i)))
			else:
				filenames2.append(parse_function(filename+"/training/%s/%s/frame_00%d.png" % (data_name,sub,i)))				
		for i in range(low,up):
			if i < 10:
				#ground_truth_flow.append(filename+"/training/flow_viz/%s/frame_000%d.png" % (sub,i))
				ground_truth_flow.append(read_flo(filename+"/training/flow/%s/frame_000%d.flo" % (sub,i)))
			else:
				ground_truth_flow.append(read_flo(filename+"/training/flow/%s/frame_00%d.flo" % (sub,i)))

	print("Observations read %d. Each obsevation contains a pair of frames and the ground truth flow." % len(filenames1))

        # create list of stacked,decoded images and reconstructed intermediate frame; concat on dimension 2 (0,1 are w,h) 2 is rgb
	image_stack = []
	for image1,image2 in zip(filenames1,filenames2):
		image_stack.append(tf.concat([image1,image2], 2))
                
	# convert to dataset object 
	dataset = tf.data.Dataset.from_tensor_slices((image_stack,ground_truth_flow))
	# I believe the estimator object train method can be passed a dataset directly
	# TODO: fix batch size to appropriate amount here
	return dataset.shuffle(1000).repeat(repeat).batch(batch)

def feed_one(path):
	label = read_flo(path +'/training/flow/alley_1/frame_0018.flo')
	image1 = parse_function(path+'/training/final/alley_1/frame_0018.png')
	image2 = parse_function(path +'/training/final/alley_1/frame_0019.png')
	stack = tf.concat([image1,image2], 2)
	dataset = tf.data.Dataset.from_tensor_slices(([stack],[label]))
	return dataset.repeat(1).batch(1)



def main():
	label = read_flo('/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete/training/flow/alley_1/frame_0044.flo')
	predict = read_flo2('/Users/renzhihuang/Desktop/final_cnn/noReLU/pNfKzeChjcq5CV7gJSZuMT/alpha0_all.flo')
	zero = tf.zeros([512,1024,2])
	loss_abs = tf.losses.absolute_difference(
		labels=label,
		predictions=label,
		weights=1,
		reduction=tf.losses.Reduction.MEAN
		)
	loss = tf.reduce_sum(tf.abs(tf.subtract(predict, label)))/1024/512/2
	sess = tf.Session()
	print(loss_abs.eval(session = sess))
	print(loss.eval(session = sess))
	print(label[0,0,0].eval(session = sess))

if __name__ == '__main__':
 	main()
