import tensorflow as tf
import numpy as np
from cnn import *

def test_conv(steps):
	sess = tf.InteractiveSession()
	shape = [1, 1, 3, 1]
	conv1 = conv('conv1', shape)
	assert(conv1.get_layer_name()=='conv1')
	assert(conv1.get_layer_type()=='conv')
	assert(conv1.is_trainable()==True)
	assert(conv1.get_shape()==shape)
	conv1.initialize()
	conv1.add_variable_summaries()
	wt, bi = conv1.get_variables()
	sess.run([tf.initialize_all_variables()])
	wt, bi = sess.run([wt, bi])
	print "initialized weights:", wt
	print "initialized biases:", bi
	print '*'*32

	input = tf.placeholder(tf.float32, [4, 2, 2, 3])

	for i in range(steps):
		X_np = np.random.randn(4, 2, 2, 3)
		print '*'*16, i, '*'*16
		y = conv1.train(input)
		print y.eval(feed_dict={input:X_np})
	
	sess.close()

if __name__=='__main__':
	test_conv(3)

