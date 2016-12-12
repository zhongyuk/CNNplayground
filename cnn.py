import abc
import tensorflow as tf
import time
import numpy as np
import math

class layer(object):
	'''an abstract metaclass for CNN layers'''
	__metaclass__ = abc.ABCMeta

	# static class variable
	LAYER_TYPES = ['conv2d', 'fc', 'activation', 'batchnorm', 'dropout', 'pool2d', 'convInception']
	
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

	def __init__(self, layer_name, keep_prob=None):
		self._layer_name = layer_name
		if keep_prob is None:
			self.keep_prob = tf.placeholder(tf.float32)
		else:
			self.keep_prob = keep_prob

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return dropout.FULLNAME

	def is_trainable(self):
		return dropout.TRAINABLE

	def train(self, input, add_output_summary=True):
		with tf.name_scope(self._layer_name):
			output = tf.nn.dropout(input, self.keep_prob)
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
				self.variable_summaries(tf.get_variable("gamma"), self._layer_name+'/scale_factors')
			if self._center:
				self.variable_summaries(tf.get_variable("beta"), self._layer_name+'/offset_factors')
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

class convInception(layer):
	'''A fixed 2D conv inception layer: 1x1, 3x3, and 5x5 filters'''

	TRAINABLE = True
	FULLNAME = "2D Convolution Inception Layer"

	def __init__(self, layer_name, input_depth, output_depth, stride=1, padding='SAME'):
		self._layer_name = layer_name
		self._stride = stride
		self._padding = padding
		self._wt_initializer = tf.truncated_normal_initializer(stddev=.01)
		self._bi_initializer = tf.constant_initializer(1.0)
		self._inception = self._build_inception(input_depth, output_depth, stride, padding)

	def _build_inception(self, input_depth, output_depth, stride, padding):
		conv1x1 = conv2d('1x1', [1,1,input_depth, output_depth], stride, padding)
		conv3x3 = conv2d('3x3', [3,3,input_depth, output_depth], stride, padding)
		conv5x5 = conv2d('5x5', [5,5,input_depth, output_depth], stride, padding)
		inception = {conv1x1.get_layer_name() : conv1x1,
					 conv3x3.get_layer_name() : conv3x3,
					 conv5x5.get_layer_name() : conv5x5}
		return inception

	def get_layer_name(self):
		return self._layer_name

	def get_layer_type(self):
		return convInception.FULLNAME

	def is_trainable(self):
		return convInception.TRAINABLE

	def get_inception_unit(self, name):
		if name not in ['1x1', '3x3', '5x5']:
			raise ValueError("name has to be '1x1', '3x3', or '5x5'")
		return self._inception[name]

	def initialize(self, wt_initializer=None, bi_initializer=None):
		if wt_initializer:
			self._wt_initializer = wt_initializer
		if bi_initializer:
			self._bi_initializer = bi_initializer

		with tf.variable_scope(self._layer_name, reuse=None) as scope:
			for conv_layer in self._inception.values():
				conv_layer.initialize(None, self._wt_initializer, self._bi_initializer)

	def get_variables(self):
		variables = {}
		with tf.variable_scope(self._layer_name, reuse=True):
			for name, conv_layer in self._inception.items():
				wt, bi = conv_layer.get_variables()
				var_dict = {'weight' : wt, 'biase' : bi}
				variables[name] = var_dict
		return variables

	def add_variable_summaries(self):
		with tf.variable_scope(self._layer_name, reuse=True):
			for conv_layer in self._inception.values():
				conv_layer.add_variable_summaries()

	def train(self, input, add_output_summary=True):
		outputs = {}
		with tf.variable_scope(self._layer_name, reuse=True):
			for name, conv_layer in self._inception.items():
				outputs[name] = conv_layer.train(input, add_output_summary)
			tensor_list = list(zip(*sorted(outputs.items(), key=lambda x: x[0]))[1])
			output = tf.concat(3, tensor_list)
			return output


class keep_prob_collection(object):
	"""
	A class for grouping all keep_prob placeholders together,
	and generating feed_dict for all keep_prob placeholders together.
	The default real value setting for each keep probability placeholder is 1.0
	"""
	def __init__(self):
		super(keep_prob_collection, self).__init__() #super().__init__ in python 3
		self._names = []
		self._feed_dict = {}

	def add_keep_prob(self, name, placeholder):
		if name in self._names:
			raise ValueError("Duplicate name already exists in keep_prob_collection names.")
		setattr(self, name, placeholder)
		self._names.append(name)
		# set the default setting
		self._feed_dict[self.get_keep_prob(name)] = 1.0

	def get_keep_prob(self, name):
		return getattr(self, name)

	def set_collection_values(self, name_value_dict):
		for name, value in name_value_dict.items():
			self._feed_dict[self.get_keep_prob(name)] = value
		return self._feed_dict

	def set_value(self, name, value):
		self._feed_dict[self.get_keep_prob(name)] = value

	def get_value(self, name):
		return self._feed_dict[self.get_keep_prob(name)]

	def get_feed_dict(self):
		return self._feed_dict

	def get_all_names(self):
		return self._names

class cnn_graph(object):

	FULLNAME = 'Convolutional Neural Network Model'

	def __init__(self, input_shape, num_class):
		# for input_shape, the first dimension doesn't matter much...
		# input_shape = [-1, image_size, image_size, image_channels]
		self._graph = tf.Graph()
		self._input_shape = input_shape
		self._num_class = num_class
		# placeholders for mini_batch training and dropout keep_prob
		self.train_X = None
		self.train_y = None
		self._keep_probs = keep_prob_collection()
		self.kp_reference_feed_dict = {}
		self._all_layers = [{'layer_name'   : 'input_layer',
							 'layer_shape'  : input_shape,
							 'layer_obj'    : None}] # as a linked list

	def get_layers(self):
		return self._all_layers

	def set_kp_value(self, name, value):
		self._keep_probs.set_value(name, value)

	def get_kp_value(self, name):
		return self._keep_probs.get_value(name)

	def set_kp_collection_values(self, name_value_dict):
		return self._keep_probs.set_collection_values(name_value_dict)

	def get_kp_collection_dict(self):
		return self._keep_probs.get_feed_dict()

	def setup_data(self, batch_size, valid_X=None, valid_y=None, test_X=None, test_y=None, ):
		#test_X, test_y, valid_X, valid_y are numpy arrays, batch_size is python int
		with self._graph.as_default():
			with tf.name_scope('train_data'):
				train_X_shape = list(self._input_shape)
				train_X_shape[0] = batch_size
				self.train_X = tf.placeholder(tf.float32, shape=train_X_shape, name='train_X')
				self.train_y = tf.placeholder(tf.float32, shape=(batch_size, self._num_class), name='train_y')

			if (test_X is not None) and (test_y is not None):
				with tf.name_scope('test_data'):
					self._test_X = tf.constant(test_X, name='test_X')
					self._test_y = tf.constant(test_y, name='test_y')

			if (valid_X is not None) and (valid_y is not None):
				with tf.name_scope('valid_data'):
					self._valid_X = tf.constant(valid_X, name='valid_X')
					self._valid_y = tf.constant(valid_y, name='valid_y')

	def add_conv2d_layer(self, layer_name, filter_size, depth, 
						 wt_initializer=None, bi_initializer=None,
						 stride=1, padding='SAME', add_output_summary=True):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		wt_shape = [filter_size, filter_size, prev_layer_shape[-1], depth]
		with self._graph.as_default():
			conv_layer = conv2d(layer_name, wt_shape, stride, padding)
			conv_layer.initialize(wt_initializer=wt_initializer, bi_initializer=bi_initializer)
			if add_output_summary:
				conv_layer.add_variable_summaries()
			if padding=='SAME':
				spacial_length = math.ceil(prev_layer_shape[1]/float(stride))
			elif padding=='VALID':
				spacial_length = math.ceil((prev_layer_shape[1] - filter_size + 1.)/stride )
			output_layer_shape = [prev_layer_shape[0], spacial_length, spacial_length, depth]
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_obj'    : conv_layer})

	def add_fc_layer(self, layer_name, out_shape, wt_initializer=None, 
					bi_initializer=None, add_output_summary=True):
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
			fc_layer.initialize(wt_initializer=wt_initializer, bi_initializer=bi_initializer)
			if add_output_summary:
				fc_layer.add_variable_summaries()
			output_layer_shape = [prev_layer_shape[0], out_shape]
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_obj'    : fc_layer})

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
				spacial_length = math.ceil(prev_layer_shape[1]/float(stride)) 
			elif padding=='VALID':
				spacial_length = math.ceil((prev_layer_shape[1] - kernel_size + 1.)/stride) 
			output_layer_shape = [prev_layer_shape[0], spacial_length, spacial_length, prev_layer_shape[-1]]
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_obj'    : pool_layer})

	def add_dropout_layer(self, layer_name):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		with self._graph.as_default():
			dropout_layer = dropout(layer_name)
			self._keep_probs.add_keep_prob(layer_name, dropout_layer.keep_prob)
			self.kp_reference_feed_dict[self._keep_probs.get_keep_prob(layer_name)] = 1.0
			output_layer_shape = list(prev_layer_shape)
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_obj'    : dropout_layer})

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
									 'layer_obj'    : act_layer})

	def add_batchnorm_layer(self, layer_name, center=True,
						    scale=True, decay=.99,
						    add_output_summary=True,):
		all_layer_names = [l['layer_name'] for l in self._all_layers]
		if layer_name in all_layer_names:
			raise ValueError("layer_name already exists. Please use a different layer name!")
		prev_layer_shape = self._all_layers[-1]['layer_shape']
		depth = prev_layer_shape[-1]
		with self._graph.as_default():
			depth = prev_layer_shape[-1]
			batchnorm_layer = batchnorm(layer_name, depth, center=center, scale=scale, decay=decay)
			batchnorm_layer.initialize()
			if add_output_summary:
				batchnorm_layer.add_variable_summaries()
			output_layer_shape = list(prev_layer_shape)
			self._all_layers.append({'layer_name'   : layer_name,
									 'layer_shape'  : output_layer_shape,
									 'layer_obj'    : batchnorm_layer})

	def __compute_logits(self, input_X, is_training, add_output_summary):
		for layer_dict in self._all_layers[1:]:
			layer_obj = layer_dict['layer_obj']
			layer_type = layer_obj.get_layer_type()
			if layer_type =='Batch Normalization Layer':
				input_X = layer_obj.train(input_X, is_training, add_output_summary)
			else:
				if layer_type=='Fully Connected Layer' and tf.rank(input_X)!=2:
					shape = input_X.get_shape().as_list()
					input_X = tf.reshape(input_X, [shape[0], -1])
				input_X = layer_obj.train(input_X, add_output_summary)
		logits = input_X
		return logits

	def __compute_cross_entropy(self, input_X, input_y, is_training, add_output_summary):
		logits = self.__compute_logits(input_X, is_training, add_output_summary)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, input_y)
		return cross_entropy

	def __compute_l2_reg(self):
		l2_reg = 0
		for layer_dict in self._all_layers[1:]:
			layer_obj = layer_dict['layer_obj']
			if layer_obj.get_layer_type()=='Fully Connected Layer':
				wt, bi = layer_obj.get_variables()
				l2_reg += tf.nn.l2_loss(wt) + tf.nn.l2_loss(bi)
		return l2_reg

	def compute_train_loss(self, l2_reg=False, l2_reg_factor=None, add_output_summary=True):
		if l2_reg and (l2_reg_factor is None):
			raise ValueEerror("with l2_reg=True l2_reg_factor cannot be None!")
		if (self.train_X is None) or (self.train_y is None):
			raise ValueError("No training data setup!")
		with self._graph.as_default():
			with tf.name_scope("train_loss"):
				cross_entropy = self.__compute_cross_entropy(self.train_X, self.train_y, True, add_output_summary)
				if l2_reg:
					l2_loss = l2_reg_factor * self.__compute_l2_reg()
				else:
					l2_loss = 0
				train_loss = tf.reduce_mean(cross_entropy + l2_loss)
			if add_output_summary:
				tf.scalar_summary('train_loss', train_loss)
			return train_loss

	def compute_valid_loss(self, add_output_summary=True):
		if (self._valid_X is None) or (self._valid_y is None):
			raise ValueError("No validation data setup!")
		with self._graph.as_default():
			valid_X = self._valid_X
			valid_y = self._valid_y
			with tf.name_scope("valid_loss"):
				cross_entropy = self.__compute_cross_entropy(valid_X, valid_y, False, add_output_summary=False)
				valid_loss = tf.reduce_mean(cross_entropy)
			if add_output_summary:
				tf.scalar_summary('valid_loss', valid_loss)
			return valid_loss

	def __compute_accuracy(self, predictions, labels):
		correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
		return accuracy

	def make_prediction(self, input_X_np):
		with self._graph.as_default():
			tf_X = tf.constant(input_X_np)
			logits = self.__compute_logits(tf_X, False, add_output_summary=False)
			predictions = tf.argmax(tf.nn.softmax(logits), 1)
			return predictions

	def evaluation(self, dataset="train", add_output_summary=True):
		#specify evaluate on "train", "valid" or "test" set
		if dataset=="train":
			if (self.train_X is None) or (self.train_y is None):
				raise ValueError("No training data setup!")
			input_X = self.train_X
			input_y = self.train_y
			is_training = True
		elif dataset=="valid":
			if (self._valid_X is None) or (self._valid_y is None):
				raise ValueError("No validation data setup!")
			input_X = self._valid_X
			input_y = self._valid_y
			is_training = False
		elif dataset=="test":
			if (self._test_X is None) or (self._test_y is None):
				raise ValueError("No test data setup!")
			input_X = self._test_X
			input_y = self._test_y
			is_training = False
			add_output_summary = False
		else:
			raise ValueError("dataset has to be 'train', 'valid', or 'test'!")
		with self._graph.as_default():
			with tf.name_scope(dataset+"_accuracy"):
				logits = self.__compute_logits(input_X, is_training, add_output_summary=False)
				predictions =  tf.nn.softmax(logits)
				accuracy = self.__compute_accuracy(predictions, input_y)
			if add_output_summary:
				tf.scalar_summary(dataset+'_accuracy', accuracy)
			return accuracy

	def setup_learning_rate(self, init_lr, exp_decay=False, decay_steps=None, 
							decay_rate=None, staircase=False, name=None,
							add_output_summary=True):
		if exp_decay and decay_steps==None and decay_rate==None:
			raise ValueError("with exp_decay=True decay_steps and decay_rate cannot be None!")
		with self._graph.as_default():
			with tf.name_scope("learning_rate"):
				if exp_decay:
					self._global_step = tf.Variable(0)
					self._learning_rate = tf.train.exponential_decay(init_lr, self._global_step, 
									  	decay_steps, decay_rate, staircase, name)
				else:
					self._global_step = tf.Variable(0)
					self._learning_rate = init_lr
			if add_output_summary:
				tf.scalar_summary("learning_rate", self._learning_rate)

	def get_learning_rate(self):
		with self._graph.as_default():
			return self._learning_rate

	def setup_optimizer(self, optimizer, l2_reg=False, l2_reg_factor=None, add_output_summary=True):
		train_loss = self.compute_train_loss(l2_reg, l2_reg_factor, add_output_summary)
		with self._graph.as_default():
			self._optimizer = optimizer(self._learning_rate).minimize(train_loss, 
							  global_step=self._global_step)
		return self._optimizer

	def merge_summaries(self):
		with self._graph.as_default():
			merged = tf.merge_all_summaries()
  			return merged

	def get_graph(self):
		return self._graph


if __name__=='__main__':
	pass




		
