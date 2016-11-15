import abc
import tensorflow as tf
import time
import numpy as np

class layer(object):
	'''an abstract metaclass for CNN layers'''
	__metaclass__ = abc.ABCMeta

	# static class variable
	LAYER_TYPES = ['conv2d', 'fc', 'activation', 'batchnorm', 'dropout', 'pool2d']
	
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


class conv2d(layer):

	TRAINABLE = True
	FULLNAME = "2D Convolution Layer"

	def __init__(self, layer_name, shape=None, stride=1, padding='SAME'):
		self._layer_name = layer_name
		if shape:
			if len(shape)!=4:
				error_msg = "The shape for " + conv2d.__name__ + \
				" weights have to be 4D, got " + str(len(shape)) + "D instead"
				raise ValueError(error_msg)
		self._shape = shape
		self._stride = stride
		self._padding = padding
		self._wt_initializer = tf.truncated_normal_initializer(stddev=.01)
		self._bi_initializer = tf.constant_initializer(1.0)

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return conv2d.FULLNAME

	def is_trainable(self):
		return conv2d.TRAINABLE

	def get_shape(self):
		if self._shape:
			return self._shape
		else:
			return "kernel_size is undefined!"

	def set_shape(self, shape):
		if len(shape)!=4:
			error_msg = "The shape for ", + conv2d.__name__ + \
			" weights have to be 4D, got " + str(len(shape)) + "D instead"
			raise ValueError(error_msg)
		else:
			self._shape = shape

	def initialize(self, shape=None, wt_initializer=None, bi_initializer=None):
		"""initialize weights and biases based on given initializers.
		If initializers are not given, using default initializers created in the constructor
		"""
		# Make sure the shape of the conv layer is define
		if shape:
			if len(shape)!=4:
				error_msg = "The shape for " + conv2d.__name__ + \
				" weights have to be 4D, got " + str(len(shape)) + "D instead"
				raise ValueError(error_msg)
			else:
				self._shape = shape
		else:
			if not self._shape:
				raise ValueError("shape is undefined! Must define shape to initalize!")

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

	def train(self, input, add_output_summary=True):
		'''input should be a tensor'''
		with tf.variable_scope(self._layer_name, reuse=True):
			wt = tf.get_variable('weights')
			bi = tf.get_variable('biases')
			with tf.name_scope('postConv'):
				output = tf.nn.conv2d(input, wt, [1, self._stride, self._stride, 1], padding=self._padding) + bi
				if add_output_summary:
					tf.histogram_summary(self._layer_name+'/postConv', output)
				return output


class fc(layer):

	TRAINABLE = True
	FULLNAME = "Fully Connected Layer"

	def __init__(self, layer_name, shape=None):
		self._layer_name = layer_name
		if shape:
			if len(shape)!=2:
				error_msg = "The shape for " + fc.__name__ + \
				" weights have to be 2D, got " + str(len(shape)) + "D instead"
				raise ValueError(error_msg)
		self._shape = shape
		self._wt_initializer = tf.truncated_normal_initializer(stddev=.01)
		self._bi_initializer = tf.constant_initializer(1.0)

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return fc.FULLNAME

	def is_trainable(self):
		return fc.TRAINABLE

	def get_shape(self):
		if self._shape:
			return self._shape
		else:
			return "shape is undefined!"

	def set_shape(self, shape):
		if len(shape)!=2:
			error_msg = "The shape for " + fc.__name__ + \
			" weights have to be 2D, got " + str(len(shape)) + "D instead"
			raise ValueError(error_msg)
		else:
			self._shape = shape

	def initialize(self, shape=None, wt_initializer=None, bi_initializer=None):
		"""initialize weights and biases based on given initializers.
		If initializers are not given, using default initializers created in the constructor
		"""
		# Make sure the shape of the conv layer is define
		if shape:
			if len(shape)!=2:
				error_msg = "The shape for " + fc.__name__ + \
				" weights have to be 2D, got " + str(len(shape)) + "D instead"
				raise ValueError(error_msg)
			else:
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

	def train(self, input, add_output_summary=True):
		'''input should be a tensor'''
		with tf.variable_scope(self._layer_name, reuse=True):
			wt = tf.get_variable('weights')
			bi = tf.get_variable('biases')
			with tf.name_scope('postFC'):
				output = tf.matmul(input, wt) + bi
				if add_output_summary:
					tf.histogram_summary(self._layer_name+'/postFC', output)
				return output


class activation(layer):

	TRAINABLE = False
	FULLNAME = "Activation Layer"

	def __init__(self, layer_name, act_func=None):
		self._layer_name = layer_name
		self._act_func = act_func

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return activation.FULLNAME

	def is_trainable(self):
		return activation.TRAINABLE

	def get_act_func(self):
		if self._act_func is not None:
			return self._act_func.__name__
		else:
			print "Activation function is undefined!"

	def set_act_func(self, act_func):
		self._act_func = act_func

	def train(self, input, add_output_summary=True):
		'''input should be a tensor''' 
		with tf.name_scope(self._layer_name):
			output = self._act_func(input)
			if add_output_summary:
				tf.histogram_summary(self._layer_name, output)
			return output

class pool2d(layer):

	TRAINABLE = False
	FULLNAME = "2D Pooling Layer"

	def __init__(self, layer_name, pool_func=None, ksize=None, strides=None, padding=None):
		self._layer_name = layer_name
		self._pool_func = pool_func
		self._ksize = ksize
		self._strides = strides
		self._padding = padding


	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return pool2d.FULLNAME

	def is_trainable(self):
		return pool2d.TRAINABLE

	def get_pool_func(self):
		if self._pool_func is not None:
			return self._pool_func.__name__
		else:
			print "Pooling function is undefined!"

	def set_pool_func(self, pool_func, ksize, strides, padding):
		self._pool_func = pool_func
		self._ksize = ksize
		self._strides = strides
		self._padding = padding

	def train(self, input, add_output_summary=True):
		with tf.name_scope(self._layer_name):
			output = self._pool_func(input, self._ksize, self._strides, self._padding)
			if add_output_summary:
				tf.histogram_summary(self._layer_name, output)
			return output

class dropout(layer):

	TRAINABLE = False
	FULLNAME = "Dropout Layer"

	def __init__(self, layer_name, keep_prob):
		self._layer_name = layer_name
		self._keep_prob = keep_prob

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return dropout.FULLNAME

	def is_trainable(self):
		return dropout.TRAINABLE

	def get_keep_prob(self):
		return self._keep_prob

	def set_keep_prob(self, keep_prob):
		self._keep_prob = keep_prob

	def train(self, input, is_training, add_output_summary=True):
		with tf.name_scope(self._layer_name):
			if is_training:
				output = tf.nn.dropout(input, self._keep_prob)
			else:
				output = input
			if add_output_summary:
				tf.histogram_summary(self._layer_name, output)
			return output

class batchnorm(layer):

	TRAINABLE = True
	FULLNAME = "Batch Normalization Layer"

	def __init__(self, layer_name, depth, center=True, scale=True, decay=.99):
		# The more data, set decay closer to 1.
		self._layer_name = layer_name
		self._depth = depth
		self._center = center
		self._scale = scale
		self._decay = decay

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return batchnorm.FULLNAME

	def is_trainable(self):
		return batchnorm.TRAINABLE

	def initialize(self):
		with tf.variable_scope(self._layer_name, reuse=None) as scope:
			if self._center:
				beta = tf.get_variable("offset_factor", self._depth, 
					    initializer=tf.constant_initializer(0.0),
						trainable=True)
			if self._scale:
				gamma = tf.get_variable("scale_factor", self._depth, 
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
				beta = tf.get_variable("offset_factor")
			if self._scale:
				gamma = tf.get_variable("scale_factor")
			moving_avg = tf.get_variable("moving_mean")
			moving_var = tf.get_variable("moving_variance")
			return beta, gamma, moving_avg, moving_var

	def add_variable_summaries(self):
		with tf.variable_scope(self._layer_name, reuse=True):
			if self._scale:
				self.variable_summaries(tf.get_variable("scale_factor"), self._layer_name+'/scale_factors')
			if self._center:
				self.variable_summaries(tf.get_variable("offset_factor"), self._layer_name+'/offset_factors')
			self.variable_summaries(tf.get_variable("moving_mean"), self._layer_name+'/moving_means')
			self.variable_summaries(tf.get_variable("moving_variance"), self._layer_name+'/moving_variances')

	def train(self, input, is_training, add_output_summary=True):
		# is_training can be tf.placeholder(tf.bool) or Python bool
		output = tf.contrib.layers.batch_norm(input, decay=self._decay, is_training=is_training, 
											  center=self._center, scale=self._scale, 
											  updates_collections=None, 
											  scope=self._layer_name, reuse=True)
		if add_output_summary:
				tf.histogram_summary(self._layer_name, output)
		return output

class build_cnn(object):

	FULLNAME = 'Convolutional Neural Network Model'

	def __init__(self, input_shape, num_class):
		self._graph = tf.Graph()
		#self._input_shape = input_shape
		self._num_class = num_class
		self._all_layers = [{'layer_name'   : 'input_layer',
							 'layer_shape'  : input_shape,
							 'layer_pointer': None}] # as a linked list

	def get_layers(self):
		return self._all_layers

	def setup_data(self, test_X, test_y, valid_X=None, valid_y=None):
		with self._graph.as_default():
			with tf.name_scope('train_data'):
				train_X_shape = [self._batch_size, self._image_size, self._image_size, self._init_depth]
				self._train_X = tf.placeholder(tf.float32, shape=train_X_shape, name='train_X')
				self._train_y = tf.placeholder(tf.float32, shape=(self._batch_size, self._num_class), name='train_y')

			with tf.name_scope('test_data'):
				self._test_X = tf.constant(test_X, name='test_X')
				self._test_y = tf.constant(test_y, name='test_y')

			if (valid_X != None) and (valid_y != None):
				with tf.name_scope('valid_data'):
					self._valid_X = tf.constant(valid_X, name='valid_X')
					self._valid_y = tf.constant(valid_y, name='valid_y')

	def add_conv2d_layer(self, layer_name, filter_size, depth, 
						 wt_initializer=None, bi_initializer=None,
						 stride=1, padding='SAME'):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		wt_shape = [filter_size, filter_size, prev_layer_shape[-1], depth]
		with self._graph.as_default():
			conv_layer = conv2d(layer_name, wt_shape, stride, padding)
			conv_layer.initialize(wt_initializer=wt_initializer, bi_initializer=bi_initializer)
			conv_layer.add_variable_summaries()
			if padding=='SAME':
				spacial_length = prev_layer_shape[1]
			elif padding=='VALID':
				spacial_length = (prev_layer_shape[1] - filter_size)//stride + 1
			output_layer_shape = [prev_layer_shape[0], spacial_length, spacial_length, depth]
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_pointer': conv_layer})

	def add_fc_layer(self, layer_name, out_shape, wt_initializer=None, bi_initializer=None):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		if len(prev_layer_shape)==4:
			in_shape = prev_layer_shape[1]*prev_layer_shape[2]*prev_layer_shape[3]
		elif len(prev_layer_shape)==2:
			in_shape = prev_layer_shape[1]
		wt_shape = [in_shape, out_shape]
		with self._graph.as_default():
			fc_layer = fc(layer_name, wt_shape)
			fc_layer.initializer(wt_initializer=wt_initializer, bi_initializer=bi_initializer)
			fc_layer.add_variable_summaries()
			output_layer_shape = [prev_layer_shape[0], out_shape]
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_pointer': fc_layer})

	def add_pool_layer(self, layer_name, pool_func=tf.nn.max_pool,
					   kernel_size=2, stride=2, padding='SAME'):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		with self._graph.as_default():
			ksize = [1, kernel_size, kernel_size, 1]
			strides = [1, stride, stride, 1]
			pool_layer = pool2d(layer_name, pool_func, ksize, strides, padding)
			if padding=='SAME':
				spacial_length = prev_layer_shape[1]//stride + 1
			elif padding=='VALID':
				spacial_length = (prev_layer_shape[1] - kernel_size)//stride + 1
			output_layer_shape = [prev_layer_shape[0], spacial_length, spacial_length, prev_layer_shape[-1]]
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_pointer': pool_layer})

	def add_dropout_layer(self, layer_name, keep_prob):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		with self._graph.as_default():
			dropout_layer = dropout(layer_name, keep_prob)
			output_layer_shape = list(prev_layer_shape)
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_pointer': dropout_layer})

	def add_act_layer(self, layer_name, act_func=tf.nn.relu):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		with self._graph.as_default():
			act_layer = activation(layer_name, act_func)
			output_layer_shape = list(prev_layer_shape)
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_pointer': act_layer})

	def add_batchnorm_layer(self, layer_name, depth, center=True,
						    scale=True, decay=.99):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		with self._graph.as_default():
			depth = prev_layer_shape[-1]
			batchnorm_layer = batchnorm(layer_name, depth, center=center, scale=scale, decay=decay)
			batchnorm_layer.initialize()
			batchnorm_layer.add_variable_summaries()
			output_layer_shape = list(prev_layer_shape)
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_pointer': batchnorm_layer})





if __name__=='__main__':
	pass




		
