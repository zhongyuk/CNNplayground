import abc
import tensorflow as tf


class layer(object):
	'''an abstract metaclass for CNN layers'''
	__metaclass__ = abc.ABCMeta

	# static class variable
	LAYER_TYPES = ['conv', 'fc', 'activation', 'batch_norm', 'dropout', 'pool']
	
	@abc.abstractmethod
	def get_layer_name(self):
		'''return a string - the name of the layer'''
		pass

	@abc.abstractmethod
	def get_layer_type(self):
		'''return a string - the type of the subclass layer - should be one of LAYER_TYPES'''
		pass

	@abc.abstractmethod
	def is_trainable(self):
		'''return a boolean - indicates if the layer is trainable'''
		pass

	@abc.abstractmethod
	def train(self):
		'''return y - resulted tensor after computation/training'''
		pass


	@classmethod
	def variable_summaries(self, var, name):
	    with tf.name_scope('summaries'):
	        avg = tf.reduce_mean(var)
	        tf.scalar_summary('mean/'+name, avg)
	        with tf.name_scope('std'):
	            std = tf.sqrt(tf.reduce_mean(tf.square(var - avg)))
	        tf.scalar_summary('std/'+name, std)
	        tf.scalar_summary('max/'+name, tf.reduce_max(var))
	        tf.scalar_summary('min/'+name, tf.reduce_min(var))
	        tf.histogram_summary(name, var)


class conv(layer):

	TRAINABLE = True

	def __init__(self, layer_name, shape=None):
		#layer.__init__(self)
		#self._graph = graph
		self._layer_name = layer_name
		self._shape = shape
		self._wt_initializer = tf.truncated_normal_initializer(stddev=.01)
		self._bi_initializer = tf.constant_initializer(1.0)


	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return conv.__name__

	def is_trainable(self):
		return conv.TRAINABLE

	def get_shape(self):
		if self._shape:
			return self._shape
		else:
			return "shape is undefined!"

	def initialize(self, shape=None, wt_initializer=None, bi_initializer=None):
		"""initialize weights and biases based on given initializers.
		If initializers are not given, using default initializers created in the constructor
		"""
		# Make sure the shape of the conv layer is define
		if shape:
			self._shape = shape
		else:
			if not self._shape:
				raise ValueError("shape is undefined! Must define shape to initalize conv layer!")

		# Use the initializers passed along by the users instead of the default initializers if they are given
		if wt_initializer:
			self._wt_initializer = wt_initializer

		if bi_initializer:
			self._bi_initializer = bi_initializer

		with tf.variable_scope(self._layer_name, reuse=None) as scope:
			wt = tf.get_variable('weights', self._shape, initializer=self._wt_initializer)
			bi = tf.get_variable('biases', self._shape[-1], initializer=self._bi_initializer)
			scope.reuse_variables()

	def get_variables(self):
		with tf.variable_scope(self._layer_name, reuse=True):
			wt = tf.get_variable('weights')
			bi = tf.get_variable('biases')
		return wt, bi

	def add_variable_summaries(self):
		with tf.variable_scope(self._layer_name, reuse=True):
			self.variable_summaries(tf.get_variable("weights"), self._layer_name+'/weights')
			self.variable_summaries(tf.get_variable("biases"), self._layer_name+'/biases')

	def train(self, input, stride=1, padding='SAME', add_output_summary=True):
		'''input should be a tensor'''
		with tf.variable_scope(self._layer_name, reuse=True):
			wt = tf.get_variable('weights')
			bi = tf.get_variable('biases')
			with tf.name_scope('postConv'):
				output = tf.nn.conv2d(input, wt, [1, stride, stride, 1], padding=padding) + bi
				if add_output_summary:
					tf.histogram_summary(self._layer_name+'/postConv', output)
				return output

if __name__=='__main__':
	pass




		
