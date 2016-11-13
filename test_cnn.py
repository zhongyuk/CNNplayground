import tensorflow as tf
import numpy as np
from cnn import *

def test_conv2d(steps):
	try:
		convx = conv2d('convx', [1,2])
	except ValueError:
		print "Successfully catch shape error"

	sess = tf.InteractiveSession()
	shape = [1, 1, 3, 1]
	conv1 = conv2d('conv1', shape) 
	assert(conv1.get_layer_name()=='conv1')
	assert(conv1.get_layer_type()=='2D Convolution Layer')
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
	assert(fc1.get_layer_type()=='Fully Connected Layer')
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

def test_act(steps):
	sess = tf.InteractiveSession()
	act1 = activation('act1')
	assert(act1.get_layer_name()=='act1')
	assert(act1.get_layer_type()=='Activation Layer')
	assert(act1.is_trainable()==False)
	act1.get_act_func()
	act1.set_act_func(tf.nn.relu)
	assert(act1.get_act_func()=='relu')

	input = tf.placeholder(tf.float32, [2, 2])
	for i in range(steps):
		X_np = np.random.randn(2,2)
		print '*'*16, i, '*'*16
		print X_np
		print "-"*32
		y = act1.train(input)
		print y.eval(feed_dict={input:X_np})
	sess.close()

def test_pool2d(steps):
	sess = tf.InteractiveSession()
	pool1 = pool2d('pool1')
	assert(pool1.get_layer_name()=='pool1')
	assert(pool1.get_layer_type()=='2D Pooling Layer')
	assert(pool1.is_trainable()==False)
	pool1.get_pool_func()
	pool1.set_pool_func(tf.nn.max_pool)
	assert(pool1.get_pool_func()=='max_pool')

	input = tf.placeholder(tf.float32, [1,2,2,1])
	for i in range(steps):
		X_np = np.random.randn(1,2,2,1)
		print '*'*16, i, '*'*16
		print X_np
		print '-'*32
		y = pool1.train(input, True, ksize=[1,2,2,1], \
			strides=[1,2,2,1], padding='SAME')
		print y.eval(feed_dict={input : X_np})

def test_dropout(steps):
	sess = tf.InteractiveSession()
	dropout1 = dropout('dropout1', 0.5)
	assert(dropout1.get_layer_name()=='dropout1')
	assert(dropout1.get_layer_type()=='Dropout Layer')
	assert(dropout1.is_trainable()==False)
	assert(dropout1.get_keep_prob()==0.5)
	dropout1.set_keep_prob(.7)
	assert(dropout1.get_keep_prob()==.7)

	input = tf.placeholder(tf.float32, [5, 2])
	for i in range(steps):
		X_np = np.random.randn(5,2)
		print "*"*16, i, "*"*16
		print X_np
		print '-'*32
		y = dropout1.train(input)
		print y.eval(feed_dict={input : X_np})
if __name__=='__main__':
	test_conv_bool = raw_input("Test conv2d layer? [y] or [n]")
	if test_conv_bool=='y':
		test_conv2d(1)

	test_fc_bool = raw_input("Test fc layer? [y] or [n]")
	if test_fc_bool=='y':
		test_fc(2)

	test_act_bool = raw_input("Test activation layer? [y] or [n]")
	if test_act_bool=='y':
		test_act(2)

	test_pool_bool = raw_input("Test pool2d layer? [y] or [n]")
	if test_pool_bool=='y':
		test_pool2d(1)

	test_dropout_bool = raw_input("Test dropout layer? [y] or [n]")
	if test_dropout_bool=='y':
		test_dropout(2)
