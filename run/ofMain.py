import tensorflow as tf
from data_read import *
from flownet_simple import *

def main():
 	# seting the path and dataset
 	filename = ("/home/snazyman/machine_learning/optical-flow/data/sintel")
 	data = "albedo"

 	# read the data
 	input = get_data(filename,data)

 	#iterators over the dataset
 	iterator = input.make_one_shot_iterator()
 	one_element = iterator.get_next()
 	with tf.Session() as sess:
 		for i in range(5):
 			print(type(one_element[0]))
 			print(one_element[0].eval().shape)
 	#path = "/Users/renzhihuang/Desktop/CIS520/project/tensorflow/data/MPI-Sintel-complete/training/albedo/alley_1/frame_0001.png"
 	#_parse_function(path)

if __name__ == '__main__':
    main()
