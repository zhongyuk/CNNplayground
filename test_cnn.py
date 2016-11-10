import tensorflow as tf
import numpy as np
from cnn import *

def test_conv(steps):
	try:
		convx = conv('convx', [1,2])
	except ValueError:
		print "Successfully catch shape error"

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
	wt_val, bi_val = sess.run([wt, bi])
	print "initialized weights:", wt_val
	print "initialized biases:", bi_val

	input = tf.placeholder(tf.float32, [4, 2, 2, 3])
	for i in range(steps):
		X_np = np.random.randn(4, 2, 2, 3)
		print '*'*16, i, '*'*16
		y = conv1.train(input)
		print y.eval(feed_dict={input:X_np})
	sess.close()

def test_fc(steps):
	try:
		fcx = fc('fcx', [1,2,3,4])
	except ValueError:
		print "Successfully catch shape error"

	sess = tf.InteractiveSession()
	shape = [2, 1]
	fc1 = fc('fc1', shape)
	assert(fc1.get_layer_name()=='fc1')
	assert(fc1.get_layer_type()=='fc')
	assert(fc1.is_trainable()==True)
	assert(fc1.get_shape()==shape)
	fc1.initialize()
	fc1.add_variable_summaries()
	wt, bi = fc1.get_variables()
	sess.run(tf.initialize_all_variables())
	wt_val, bi_val = sess.run([wt, bi])
	print "initialized weights: ", wt_val
	print "initialized biases: ", bi_val

	input = tf.placeholder(tf.float32, [1, 2])
	for i in range(steps):
		X_np = np.random.randn(1, 2)
		print '*'*16, i, '*'*16
		y = fc1.train(input)
		print y.eval(feed_dict={input:X_np})
	sess.close()

if __name__=='__main__':
	test_conv_bool = raw_input("Test conv layer? [y] or [n]")
	if test_conv_bool=='y':
		test_conv(1)

	test_fc_bool = raw_input("Test fc layer? [y] or [n]")
	if test_fc_bool=='y':
		test_fc(2)

