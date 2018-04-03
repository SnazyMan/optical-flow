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
			train_labels = np.resize(data, (h, w, 2))
			train_labels = tf.convert_to_tensor(train_labels)
	return train_labels

def parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  #image_resized2 = tf.image.resize_images(image_decoded2, [28, 28])
  return image_decoded

def get_data(filename,data_name):
	''' filename: the path of MPI-Sintel-complete
	    data_name: the dataset we use (albedo, clean or ...)'''

	# get the sub directory for this dataset
	path = filename+"/training/"+data_name
	subdir = next(os.walk(path))[1]

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
		for i in range(1,number):
			if i < 10:
				filenames1.append(parse_function(filename+"/training/%s/%s/frame_000%d.png" % (data_name,sub,i)))
			else:
				filenames1.append(parse_function(filename+"/training/%s/%s/frame_00%d.png" % (data_name,sub,i)))
		for i in range(2,number+1):
			if i < 10:
				filenames2.append(parse_function(filename+"/training/%s/%s/frame_000%d.png" % (data_name,sub,i)))
			else:
				filenames2.append(parse_function(filename+"/training/%s/%s/frame_00%d.png" % (data_name,sub,i)))				
		for i in range(1,number):
			if i < 10:
				#ground_truth_flow.append(filename+"/training/flow_viz/%s/frame_000%d.png" % (sub,i))
				ground_truth_flow.append(read_flo(filename+"/training/flow/%s/frame_000%d.flo" % (sub,i)))
			else:
				ground_truth_flow.append(read_flo(filename+"/training/flow/%s/frame_00%d.flo" % (sub,i)))

	# create the dataset
	print("Observations read %d. Each obsevation contains a pair of frames and the ground truth flow." % len(filenames1))
	# create the dataset, every instance is a tensor obj after this
	dataset =tf.data.Dataset.from_tensor_slices((filenames1,filenames2,ground_truth_flow))

	return dataset

# def main():
# 	# seting the path and dataset
# 	filename = ("/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete")
# 	data = "albedo"

# 	# read the data
# 	input = get_data(filename,data)

# 	#iterators over the dataset
# 	iterator = input.make_one_shot_iterator()
# 	one_element = iterator.get_next()
# 	with tf.Session() as sess:
# 		for i in range(5):
# 			print(type(one_element[0]))
# 			print(one_element[0].eval().shape)
# 	#path = "/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete/training/albedo/alley_1/frame_0001.png"
# 	#_parse_function(path)


# if __name__ == '__main__':
# 	main()