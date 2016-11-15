import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.framework import ops
from tf_batchnorm import batch_norm

sys.path.append("../")
from cnn import layer

class batchnorm_test(layer):

	TRAINABLE = True
	FULLNAME = "Batch Normalization Layer"

	def __init__(self, layer_name, depth, center=True, scale=True, decay=.99):
		# The more data, set decay to be closer to 1.
		self._layer_name = layer_name
		self._depth = depth
		self._center = center
		self._scale = scale
		self._decay = decay

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return batchnorm_test.FULLNAME

	def is_trainable(self):
		return batchnorm_test.TRAINABLE

	def initialize(self):
		with tf.variable_scope(self._layer_name, reuse=None) as scope:
			if self._center:
				beta = tf.get_variable("beta", self._depth, 
					    initializer=tf.constant_initializer(0.2),
						trainable=True)
			if self._scale:
				gamma = tf.get_variable("gamma", self._depth, 
						initializer=tf.constant_initializer(.5),
						trainable=True)
			moving_avg = tf.get_variable("moving_mean", self._depth,
						 initializer=tf.constant_initializer(0.0),
						 trainable=False)
			moving_var = tf.get_variable("moving_variance", self._depth,
						 initializer=tf.constant_initializer(1.0),
						 trainable=False)
			scope.reuse_variables()

	def get_variables(self):
		with tf.variable_scope(self._layer_name, reuse=True) as scope:
			if self._center:
				beta = tf.get_variable("beta")
			if self._scale:
				gamma = tf.get_variable("gamma")
			moving_avg = tf.get_variable("moving_mean")
			moving_var = tf.get_variable("moving_variance")
			return beta, gamma, moving_avg, moving_var

	def add_variable_summaries(self):
		with tf.variable_scope(self._layer_name, reuse=True):
			if self._scale:
				self.variable_summaries(tf.get_variable("gamma"), self._layer_name+'/gammas')
			if self._center:
				self.variable_summaries(tf.get_variable("beta"), self._layer_name+'/betas')
			self.variable_summaries(tf.get_variable("moving_mean"), self._layer_name+'/moving_means')
			self.variable_summaries(tf.get_variable("moving_variance"), self._layer_name+'/moving_variances')

	def train(self, input, is_training, add_output_summary=True):
		output = batch_norm(input, decay=self._decay, is_training=is_training, center=self._center, 
							scale=self._scale, updates_collections=None, 
							scope=self._layer_name, reuse=True)
		if add_output_summary:
				tf.histogram_summary(self._layer_name, output)
		return output

def test_batchnorm(steps):
	sess = tf.InteractiveSession()
	batchnorm1 = batchnorm_test('batchnorm1', 3)
	assert(batchnorm1.get_layer_name()=='batchnorm1')
	assert(batchnorm1.get_layer_type()=='Batch Normalization Layer')
	assert(batchnorm1.is_trainable()==True)
	batchnorm1.initialize()
	batchnorm1.add_variable_summaries()
	sess.run([tf.initialize_all_variables()])
	beta, gamma, moving_avg, moving_var = batchnorm1.get_variables()
	beta_val, gamma_val, mv_avg, mv_var = sess.run([beta, gamma, moving_avg, moving_var])
	print "initialized variables values", '='*8
	print "beta", '-'*16; print beta_val
	print "gamma", '-'*16; print gamma_val
	print "moving mean", "-"*16; print mv_avg
	print "moving variance", "-"*16; print mv_var
	print "="*32

	input = tf.placeholder(tf.float32, [2,3])
	for i in range(steps):
		X_np = (i+1)*np.ones([2,3])
		print "*"*16, i, "*"*16
		print X_np
		print '+'*16, 'is_training==True','+'*16
		y = batchnorm1.train(input, True)
		beta, gamma, moving_avg, moving_var = batchnorm1.get_variables()
		loc_avg, loc_var = tf.nn.moments(input, [0])
		with ops.control_dependencies([loc_avg, loc_var, beta, gamma]):
			y_bn = tf.nn.batch_normalization(input, loc_avg, loc_var, beta,
			   gamma, variance_epsilon=0.001)
		mv_avg, mv_var, y_val, ybn_val = sess.run([moving_avg, moving_var, y, y_bn], 
												 feed_dict={input : X_np})
		print "moving mean", "-"*16; print mv_avg
		print "moving variance", "-"*16; print mv_var
		print "output from tf_batchnorm", "-"*16; print y_val
		print "output from self developped code snippet", "-"*16; print ybn_val

		print '+'*16, 'is_training==False','+'*16
		y = batchnorm1.train(input, False)
		beta, gamma, moving_avg, moving_var = batchnorm1.get_variables()
		with ops.control_dependencies([beta, gamma, moving_avg, moving_var]):
			y_bn = tf.nn.batch_normalization(input, moving_avg, moving_var, beta,
										 	 gamma, variance_epsilon=.001)
		mv_avg, mv_var, y_val, ybn_val = sess.run([moving_avg, moving_var, y, y_bn],
										 feed_dict={input : X_np})
		print "moving mean", "-"*16; print mv_avg
		print "moving variance", "-"*16; print mv_var
		print "output from tf_batchnorm", "-"*16; print y_val
		print "output from self developped code snippet", "-"*16; print ybn_val

if __name__ == '__main__':
	steps = int(raw_input("Run how many steps?"))
	test_batchnorm(steps)