import tensorflow as tf
import numpy as np
from data_read import *

def reconstruct(flo,image):
	sess = tf.Session()
	flow_np = sess.run(flo)
	img_np = sess.run(image)
	if (np.size(flow_np,1) != np.size(img_np,1)):
		print("Flo and image doesn't match!")
	if (np.size(flow_np,0) != np.size(img_np,0)):
		print("Flo and image doesn't match!")
	img_re = img_np.copy()
	for i in range(0,np.size(flow_np,0)):
		for j in range(0,np.size(flow_np,1)):
			x = i+int(round(flow_np[i,j,1]))
			y = j+int(round(flow_np[i,j,0]))
			if (x < 0 or x >= np.size(flow_np,0)):
				continue
			if (y < 0 or y >= np.size(flow_np,1)):
				continue
			img_re[x,y,:] = img_np[i,j,:]

	return tf.convert_to_tensor(img_re)
