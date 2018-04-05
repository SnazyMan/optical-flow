import tensorflow as tf
import numpy as np
import os

import tensorflow as tf
import numpy as np
import os

def read_flo(filename):
	"""Import .flo file for labels - pass name of .flo file as argument"""
	# WARNING: this will work on little-endian architectures (eg Intel x86) only!
	with open(filename, 'rb') as f:
		magic = np.fromfile(f, np.float32, count=1)
		if 202021.25 != magic:
			print('Magic number incorrect. Invalid .flo file')
		else:
			w = np.fromfile(f, np.int32, count=1)[0]
			h = np.fromfile(f, np.int32, count=1)[0]
			# print('Reading %d x %d flo file' % (h, w))
			data = np.fromfile(f, np.float32, count=2*w*h)
			# Reshape data into 3D array (columns, rows, bands
			train_labels = np.resize(data, (512, w, 2))
			train_labels = tf.convert_to_tensor(train_labels)
	return train_labels

def parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string,channels=3)
  image_resized = tf.image.resize_images(image_decoded, [512, 1024])  
  return image_resized

def get_data(filename,data_name):
	''' filename: the path of MPI-Sintel-complete
	    data_name: the dataset we use (albedo, clean or ...)'''

	# get the sub directory for this dataset
	path = filename+"/training/"+data_name
	subdir = next(os.walk(path))[1]
	subdir.sort()

	# read only 4 sub directories
	subdir = [subdir[x] for x in range(0,5)]

	# get the list of file names
	# filename1: list of frame1 tensor
	# filename2: list of frame2 tensor
	# ground_truth_flow: list of flow tensor
	filenames1 = []
	filenames2 = []
	ground_truth_flow = [];
	for sub in subdir:
		number = len(next(os.walk(filename+"/training/"+data_name+"/"+sub))[2])
		#print(sub+":"+str(number))
		for i in range(1,number):
			filenames1.append(parse_function(filename+"/training/%s/%s/frame_%04d.png" % (data_name,sub,i)))
		for i in range(2,number+1):
			filenames2.append(parse_function(filename+"/training/%s/%s/frame_%04d.png" % (data_name,sub,i)))
		for i in range(1,number):
			#ground_truth_flow.append(filename+"/training/flow_viz/%s/frame_000%d.png" % (sub,i))
			ground_truth_flow.append(read_flo(filename+"/training/flow/%s/frame_%04d.flo" % (sub,i)))

	print("Observations read %d. Each obsevation contains a pair of frames and the ground truth flow." % len(filenames1))

        # create list of stacked,decoded images; concat on dimension 2 (0,1 are w,h) 2 is rgb
	image_stack = []
	for image1,image2 in zip(filenames1,filenames2):
		image_stack.append(tf.concat([image1,image2], 2))
                
	# convert to dataset object 
	dataset = tf.data.Dataset.from_tensor_slices((image_stack,ground_truth_flow))

        # I believe the estimator object train method can be passed a dataset directly
        # TODO: fix batch size to appropriate amount here
	return dataset.batch(10)

if __name__ == '__main__':
 	main()
