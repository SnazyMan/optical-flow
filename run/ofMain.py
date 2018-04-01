import tensorflow as tf
from data_read import *
from flownet_simple import *

def main():
 	# seting the path and dataset
 	filename = ("/sintel")
 	data = "albedo"

 	# read the data
 	input = get_data(filename,data)

        #iterators over the dataset
 	iterator = input.make_one_shot_iterator()
 	one_element = iterator.get_next()
 	with tf.Session() as sess:
            for i in range(5):
                print("hi")
                print(type(one_element[0]))
                print(type(one_element[1]))
                print(type(one_element[2]))

if __name__ == '__main__':
    main()
