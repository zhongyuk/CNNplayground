import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

# load data, prepare data
mnist = fetch_mldata('MNIST original', data_home='./')

mnist_X = mnist.data.astype(np.float32)
mnist_y = mnist.target.astype(np.float32)

# One-Hot-Encode y
num_classes = 10
mnist_y = np.arange(num_classes)==mnist_y[:, None]
mnist_y = mnist_y.astype(np.float32)

#Split training, validation, testing data
train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=10000,\
                                                      random_state=132, stratify=mnist.target)
train_X, test_X,  train_y, test_y  = train_test_split(train_X, train_y, test_size=10000,\
                                                     random_state=324, stratify=train_y)
print 'Dataset\t\tFeatureShape\tLabelShape'
print 'Training set:\t', train_X.shape,'\t', train_y.shape
print 'Validation set:\t', valid_X.shape,'\t', valid_y.shape
print 'Testing set:\t', test_X.shape, '\t', test_y.shape

#Build a simple 2 layer neural network graph
num_features = train_X.shape[1]
batch_size = 64
hidden_layer_size = 1024

# An initialization function
def initialize(scope, shape, wt_initializer, center=True, scale=True):
    with tf.variable_scope(scope, reuse=None) as sp:
        wt = tf.get_variable("weights", shape, initializer=wt_initializer)
        bi = tf.get_variable("biases", shape[-1], initializer=tf.constant_initializer(1.))
        if center:
            beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0))
        if scale:
            gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
        moving_avg = tf.get_variable("moving_mean", shape[-1], initializer=tf.constant_initializer(0.0), \
                                     trainable=False)
        moving_var = tf.get_variable("moving_variance", shape[-1], initializer=tf.constant_initializer(1.0), \
                                     trainable=False)
        sp.reuse_variables()

def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.scalar_summary('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.scalar_summary('stddev/' + name, stddev)
		tf.scalar_summary('max/' + name, tf.reduce_max(var))
		tf.scalar_summary('min/' + name, tf.reduce_min(var))
		tf.histogram_summary(name, var)

# Build Graph
init_lr = 0.001
graph = tf.Graph()
with graph.as_default():
	with tf.name_scope("input"):
		tf_X = tf.placeholder(tf.float32, shape=[None, num_features])
		tf_y = tf.placeholder(tf.float32, shape=[None, num_classes])
	
	# setup layers
	layers = [{'scope':'layer1', 'shape':[num_features, hidden_layer_size],'initializer':tf.truncated_normal_initializer(stddev=0.01)},
	{'scope':'layer2', 'shape':[hidden_layer_size, num_classes],'initializer':tf.truncated_normal_initializer(stddev=0.01)}]
	
	# initialize layers
	for layer in layers:
		initialize(layer['scope'], layer['shape'], layer['initializer'])

	def nn_layer(X, scope, is_training, decay=.9):
		with tf.variable_scope(scope, reuse=True):
			wt = tf.get_variable("weights")
			bi = tf.get_variable("biases")
			with tf.name_scope('Wx_plus_b'):
				prebn = tf.matmul(X, wt) + bi
				tf.histogram_summary(scope+'/prebn', prebn)
		postbn = tf.contrib.layers.batch_norm(prebn, decay=decay, is_training=is_training,
            									center=True, scale=False,
                                                updates_collections=None, scope=scope, reuse=True)
		with tf.name_scope(scope):
			tf.histogram_summary(scope+'/postbn', postbn)
			activations = tf.nn.relu(postbn, name="activation")
			tf.histogram_summary(scope+'/activations', activations)
			return activations

	is_training = tf.placeholder(tf.bool)
	layer1 = nn_layer(tf_X, 'layer1', is_training)

	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		dropped = tf.nn.dropout(layer1, keep_prob)

	logits = nn_layer(dropped, 'layer2', is_training)
	with tf.name_scope("loss"):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_y))
	tf.scalar_summary('loss', loss)
	
	with tf.name_scope("train"):
		global_step = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(init_lr,global_step, decay_steps=500, decay_rate=0.95, staircase=True)
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

	with tf.name_scope("accuracy"):
		correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(tf_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	tf.scalar_summary('accuracy', accuracy)

	merged = tf.merge_all_summaries()

num_steps = 500
with tf.Session(graph=graph) as sess:
	train_writer = tf.train.SummaryWriter('tmp/' + '/train',sess.graph)
	test_writer = tf.train.SummaryWriter('tmp/' + '/test')
	tf.initialize_all_variables().run()
	print("Initialized")
	for step in range(num_steps):
		if step%10==0:
			feed_dict = {tf_X : valid_X,
						tf_y : valid_y,
						keep_prob : 1.0,
						is_training : False}
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
			test_writer.add_summary(summary, step)
			print "validation accuracy: ", acc
		else:
			offset = (step * batch_size) % (train_y.shape[0] - batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			feed_dict = {tf_X : batch_X, 
						tf_y : batch_y, 
						keep_prob : 0.6,
						is_training : False}
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
			train_writer.add_summary(summary, step)
	train_writer.close()
	test_writer.close()


