import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.framework import ops
from tf_batchnorm import batch_norm
from tensorflow.python.training import moving_averages

sys.path.append("../")
from cnn import layer

class batchnorm_v1(layer):

	TRAINABLE = True
	FULLNAME = "Batch Normalization Layer"

	def __init__(self, layer_name, depth, center=True, scale=True, decay=.9):
		# The more data, set decay to be closer to 1.
		self._layer_name = layer_name
		self._depth = depth
		self._center = center
		self._scale = scale
		self._decay = decay

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return batchnorm_v1.FULLNAME

	def is_trainable(self):
		return batchnorm_v2.TRAINABLE

	def initialize(self):
		with tf.variable_scope(self._layer_name, reuse=None) as scope:
			if self._center:
				beta = tf.get_variable("beta", self._depth, 
					    initializer=tf.constant_initializer(0.0),
						trainable=True)
			if self._scale:
				gamma = tf.get_variable("gamma", self._depth, 
						initializer=tf.constant_initializer(1.0),
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

class batchnorm_v2(layer):
	TRAINABLE = True
	FULLNAME = "Batch Normalization Layer"

	def __init__(self, layer_name, depth, decay=0.99):
		self._layer_name = layer_name
		self._depth = depth
		self._decay = decay

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return batchnorm_v2.FULLNAME

	def is_trainable(self):
		return batchnorm_v2.TRAINABLE 

	def initialize(self):
		with tf.variable_scope(self._layer_name, reuse=None) as scope:
			beta = tf.get_variable("beta", self._depth, 
				    initializer=tf.constant_initializer(0.0),
					trainable=True)
			gamma = tf.get_variable("gamma", self._depth, 
					initializer=tf.constant_initializer(1.0),
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
			beta = tf.get_variable("beta")
			gamma = tf.get_variable("gamma")
			moving_avg = tf.get_variable("moving_mean")
			moving_var = tf.get_variable("moving_variance")
			return beta, gamma, moving_avg, moving_var

	def train(self, input, is_training):
		with tf.variable_scope(self._layer_name, reuse=True):
			gamma, beta = tf.get_variable("gamma"), tf.get_variable("beta")
			moving_avg, moving_var = tf.get_variable("moving_mean"), tf.get_variable("moving_variance")
			shape = input.get_shape().as_list()
			control_inputs = []
			if is_training:
				avg, var = tf.nn.moments(input, range(len(shape)-1))
				update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, self._decay)
				update_moving_var = moving_averages.assign_moving_average(moving_var, var, self._decay)
				control_inputs = [update_moving_var, update_moving_avg]
			else:
				avg = moving_avg
				var = moving_var
			with tf.control_dependencies(control_inputs):
				output = tf.nn.batch_normalization(input, avg, var, 
						 offset=beta, scale=gamma, variance_epsilon=0.001)
			return output 


def test_batchnorm(steps):
	sess = tf.InteractiveSession()
	batchnorm1 = batchnorm_v1('batchnorm1', 3)
	batchnorm2 = batchnorm_v2('batchnorm2', 3)
	assert(batchnorm1.get_layer_name()=='batchnorm1')
	assert(batchnorm1.get_layer_type()=='Batch Normalization Layer')
	assert(batchnorm1.is_trainable()==True)
	batchnorm1.initialize()
	batchnorm2.initialize()
	batchnorm1.add_variable_summaries()
	sess.run([tf.initialize_all_variables()])
	beta1, gamma1, moving_avg1, moving_var1 = batchnorm1.get_variables()
	beta2, gamma2, moving_avg2, moving_var2 = batchnorm2.get_variables()
	beta_val1, gamma_val1, mv_avg1, mv_var1 = sess.run([beta1, gamma1, moving_avg1, moving_var1])
	beta_val2, gamma_val2, mv_avg2, mv_var2 = sess.run([beta2, gamma2, moving_avg2, moving_var2])
	print "initialized variables values", '='*8
	print "beta1", '-'*16; print beta_val1
	print "beta2", '-'*16; print beta_val2
	print "gamma1", '-'*16; print gamma_val1
	print "gamma2", '-'*16; print gamma_val2
	print "moving mean1", "-"*16; print mv_avg1
	print "moving mean2", "-"*16; print mv_avg2
	print "moving variance1", "-"*16; print mv_var1
	print "moving variance2", "-"*16; print mv_var2
	print "="*32

	input = tf.placeholder(tf.float32, [2,3])
	for i in range(steps):
		#X_np = (i+1)*np.ones([2,3])
		X_np = np.random.normal(loc=1.0, scale=.5, size=[2,3])
		print "*"*16, i, "*"*16
		print X_np
		print '+'*16, 'is_training==True','+'*16
		y1 = batchnorm1.train(input, True)
		beta1, gamma1, moving_avg1, moving_var1 = batchnorm1.get_variables()
		y2 = batchnorm2.train(input, True)
		beta2, gamma2, moving_avg2, moving_var2 = batchnorm2.get_variables()
		loc_avg, loc_var = tf.nn.moments(input, [0])
		with ops.control_dependencies([loc_avg, loc_var, beta1, gamma1, beta2, gamma2]):
			y_bn1 = tf.nn.batch_normalization(input, loc_avg, loc_var, beta1,
			   gamma1, variance_epsilon=0.001)
			y_bn2 = tf.nn.batch_normalization(input, loc_avg, loc_var, beta2,
			   gamma2, variance_epsilon=0.001)
		mv_avg1, mv_var1, y_val1, mv_avg2, mv_var2, y_val2, ybn_val1, ybn_val2 = sess.run([moving_avg1, moving_var1, y1,
												 moving_avg2, moving_var2, y2, y_bn1, y_bn2], feed_dict={input : X_np})
		print "moving mean1", "-"*16; print mv_avg1
		print "moving mean2", "-"*16; print mv_avg2
		print "moving variance1", "-"*16; print mv_var1		
		print "moving variance2", "-"*16; print mv_var2
		print "output from tf_batchnorm1", "-"*16; print y_val1
		print "output from tf_batchnorm2", "-"*16; print y_val2
		print "output for verifying mean1 and variance1", "-"*16; print ybn_val1
		print "output for verifying mean2 and variance2", "-"*16; print ybn_val2

		print '+'*16, 'is_training==False','+'*16
		y1 = batchnorm1.train(input, False)
		y2 = batchnorm2.train(input, False)
		beta1, gamma1, moving_avg1, moving_var1 = batchnorm1.get_variables()
		beta2, gamma2, moving_avg2, moving_var2 = batchnorm2.get_variables()
		with ops.control_dependencies([beta1, gamma1, moving_avg1, moving_var1, beta2, gamma2, moving_avg2, moving_var2]):
			y_bn1 = tf.nn.batch_normalization(input, moving_avg1, moving_var1, beta1,
										 	 gamma1, variance_epsilon=.001)
			y_bn2 = tf.nn.batch_normalization(input, moving_avg2, moving_var2, beta2,
											 gamma2, variance_epsilon=.001)
		mv_avg1, mv_var1, y_val1, moving_avg2, moving_var2, y_val2, ybn_val1, ybn_val2 = sess.run([moving_avg1, moving_var1, y1, 
											moving_avg2, moving_var2, y2, y_bn1, y_bn2],
										 feed_dict={input : X_np})
		print "moving mean1", "-"*16; print mv_avg1
		print "moving mean2", "-"*16; print mv_avg2
		print "moving variance1", "-"*16; print mv_var1
		print "moving variance2", "-"*16; print mv_var2
		print "output from tf_batchnorm", "-"*16; print y_val1
		print "output from self devp batchnorm", "-"*16; print y_val2
		print "output for verifying moving_avg1, moving_var1", "-"*16; print ybn_val1
		print "output for verifying moving_avg2, moving_var2", "-"*16; print ybn_val2


def test_convergence(steps, decay):
	sess = tf.InteractiveSession()
	batchnorm = batchnorm_v2('batchnorm2', 3, decay=decay)
	batchnorm.initialize()
	sess.run([tf.initialize_all_variables()])

	X = tf.placeholder(tf.float32, [1, 1])
	testXnp = np.random.normal(loc=1.0, scale=0.01, size=[1,1])
	print "*"*7, "test X: ", testXnp, "*"*7
	for i in range(steps):
		trainXnp = np.random.normal(loc=1.0, scale=.01, size=[1,1])
		trainy = batchnorm.train(X, True)
		trainy.eval(feed_dict={X : trainXnp})
		testy = batchnorm.train(X, False)
		_, _, moving_avg, moving_var = batchnorm.get_variables()
		#avg_avg = tf.reduce_mean(moving_avg);avg_var = tf.reduce_mean(moving_var)
		mv_avg, mv_var, _= sess.run([moving_avg, moving_var, testy], feed_dict={X : testXnp})
		if i%50==0:
			print "+"*17, i, "+"*17
			print "moving mean: ", mv_avg
			print "moving variance: ", mv_var
		epsilon = 10**(-3)
		if (abs(mv_avg[0] - 1.0)<epsilon):
			print i


if __name__ == '__main__':
	steps = int(raw_input("Run how many steps?"))
	#test_batchnorm(steps)
	decay = float(raw_input("decay rate?"))
	test_convergence(steps, decay)