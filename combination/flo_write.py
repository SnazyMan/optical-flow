from data_read import *
import tensorflow as tf
import numpy as np
import struct
def flo_write(flo,path):
	# with tf.Session() as sess:
	# 	flo_np = sess.run(flo)
	tag = 'PIEH'
	flo_np = flo
	try:
		height = flo_np.shape[0]
	except:
		print ("no height dimension")
	try:
		width = flo_np.shape[1]
	except:
		print ("no width dimension")
	try:
		band = flo_np.shape[2]
	except:
		print ("no band dimension")
	if (band != 2):
		print ("band is not 2")
	header = struct.pack('4sii',tag.encode('ascii'),width,height)
	with open(path,'wb') as f:
		f.write(header)
		f.write(flo_np.tobytes())