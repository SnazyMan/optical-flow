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

def _parse_function(filename1,filename2,ground_truth_flow):
  image_string1 = tf.read_file(filename1)
  image_string2 = tf.read_file(filename2)
  image_decoded1 = tf.image.decode_png(image_string1)
  image_decoded2 = tf.image.decode_png(image_string2)
  #image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return (image_decoded1,image_decoded2,ground_truth_flow)

def get_data(filename,data_name):
	''' filename: the path of MPI-Sintel-complete
	    data_name: the dataset we use (albedo, clean or ...)'''

	# get the sub directory for this dataset
	path = filename+"/training/"+data_name
	subdir = next(os.walk(path))[1]

	# get the list of file names
	# filename1: list of file names for the 1st frame in a pair
	# filename2: list of file names for the 2nd frame in a pair
	filenames1 = []
	filenames2 = []
	ground_truth_flow = [];
	for sub in subdir:
		number = len(next(os.walk(filename+"/training/"+data_name+"/"+sub))[2])
		for i in range(1,number):
			if i < 10:
				filenames1.append(filename+"/training/%s/%s/frame_000%d.png" % (data_name,sub,i))
			else:
				filenames1.append(filename+"/training/%s/%s/frame_00%d.png" % (data_name,sub,i))
		for i in range(2,number+1):
			if i < 10:
				filenames2.append(filename+"/training/%s/%s/frame_000%d.png" % (data_name,sub,i))
			else:
				filenames2.append(filename+"/training/%s/%s/frame_00%d.png" % (data_name,sub,i))				
		for i in range(1,number):
			if i < 10:
				#ground_truth_flow.append(filename+"/training/flow_viz/%s/frame_000%d.png" % (sub,i))
				ground_truth_flow.append(read_flo(filename+"/training/flow/%s/frame_000%d.flo" % (sub,i)))
			else:
				ground_truth_flow.append(read_flo(filename+"/training/flow/%s/frame_00%d.flo" % (sub,i)))

	# create the dataset
	print("Observations read %d. Each obsevation contains a pair of frames and the ground truth flow" % len(filenames1))
	# create the dataset, every instance is a tensor obj after this
	dataset =tf.data.Dataset.from_tensor_slices((filenames1,filenames2,ground_truth_flow))
	# do transformation for each element (decoding the image)
	dataset  = dataset.map(_parse_function)
	return dataset

# def main():
# 	# seting the path and dataset
# 	filename = ("/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete")
# 	data = "albedo"

# 	# read the data
# 	input = get_data(filename,data)

# 	# #iterators over the dataset
# 	# iterator = input.make_one_shot_iterator()
# 	# one_element = iterator.get_next()
# 	# with tf.Session() as sess:
# 	# 	for i in range(5):
# 	# 		print("hi")
# 	# 		print(type(one_element[0]))
# 	# 		print(type(one_element[1]))
# 	# 		print(type(one_element[2]))


# if __name__ == '__main__':
# 	main()