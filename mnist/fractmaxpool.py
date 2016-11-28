import numpy as np
from utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split

train_filename = "/Users/Zhongyu/Documents/projects/kaggle/mnist/train.csv"
X, y = load_data(train_filename)
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=2000,
									random_state=263, stratify=y)
valid_X, test_X, valid_y, test_y = train_test_split(valid_X, valid_y, test_size=1000,
									random_state=932, stratify=valid_y)
train_X = reshape_data(train_X)
valid_X = reshape_data(valid_X)
test_X = reshape_data(test_X)

batch_size = 64

def initialize_withbn(scope, shape, wt_initializer, center=True, scale=True):
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

def initialize(scope, shape, wt_initializer):
    with tf.variable_scope(scope, reuse=None) as sp:
        wt = tf.get_variable("weights", shape, initializer=wt_initializer)
        bi = tf.get_variable("biases", shape[-1], initializer=tf.constant_initializer(1.))
        sp.reuse_variables()

graph = tf.Graph()
with graph.as_default():
	train_X_tf = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
	train_y_tf = tf.placeholder(tf.float32, shape=[batch_size, 10])
	valid_X_tf = tf.constant(valid_X)
	valid_y_tf = tf.constant(valid_y)
	test_X_tf = tf.constant(test_X)
	test_y_tf = tf.constant(test_y)

	wt_initializer = tf.truncated_normal_initializer(stddev=.010)
	scopes = [['conv1', [3, 3, 1, 4]], ['conv2', [3, 3, 4, 4]],
			 ['conv3', [3, 3, 4, 4]], ['conv4', [3, 3, 4, 4]],
			 ['fc1', [196, 1024]], ['fc2', [1024, 10]]] #196, 576, 256

	for scope in scopes:
		name = scope[0]
		shape = scope[1]
		initialize(name, shape, wt_initializer)

	def model_fracmaxpool(input):
		conv_scopes = ['conv1', 'conv2', 'conv3', 'conv4']
		X = input
		for conv_scope in conv_scopes:
			with tf.variable_scope(conv_scope, reuse=True):
				wt = tf.get_variable("weights")
				bi = tf.get_variable("biases")
				X = tf.nn.conv2d(X, wt, [1,1,1,1], padding='SAME') + bi
			X, row_seq, col_seq = tf.nn.fractional_max_pool(X, [1, 1.4, 1.4, 1],
				pseudo_random=True,overlapping=True, seed=82593)
			print "shape of X: ", X.get_shape().as_list()
			print "shape of row_seq: ", row_seq.get_shape().as_list()
			print "shape of col_seq: ", col_seq.get_shape().as_list()
		fc_scopes = ["fc1", "fc2"]
		shape = X.get_shape().as_list()
		X = tf.reshape(X, [shape[0], -1])
		for fc_scope in fc_scopes:
			with tf.variable_scope(fc_scope, reuse=True):
				wt = tf.get_variable("weights")
				bi = tf.get_variable("biases")
				X = tf.matmul(X, wt) + bi
		return X

	def model(input):
		conv_scopes = ['conv1', 'conv2']
		X = input
		for conv_scope in conv_scopes:
			with tf.variable_scope(conv_scope, reuse=True):
				wt = tf.get_variable("weights")
				bi = tf.get_variable("biases")
				X = tf.nn.conv2d(X, wt, [1,1,1,1], padding='SAME') + bi
			X = tf.nn.max_pool(X, [1,2,2,1],[1,2,2,1], padding='SAME')
		fc_scopes = ["fc1", "fc2"]
		shape = X.get_shape().as_list()
		X = tf.reshape(X, [shape[0], -1])
		for fc_scope in fc_scopes:
			with tf.variable_scope(fc_scope, reuse=True):
				wt = tf.get_variable("weights")
				bi = tf.get_variable("biases")
				X = tf.matmul(X, wt) + bi
		return X

	train_logits = model_fracmaxpool(train_X_tf)
	train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, train_y_tf)) 
	train_pred = tf.nn.softmax(train_logits)
	learning_rate = 0.005
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
	valid_logits = model_fracmaxpool(valid_X_tf)
	valid_pred = tf.nn.softmax(valid_logits)
	test_logits = model_fracmaxpool(test_X_tf)
	test_pred = tf.nn.softmax(test_logits)

	def compute_accuracy(pred, label):
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		return accuracy

	valid_accuracy = compute_accuracy(valid_pred, valid_y_tf)
	train_accuracy = compute_accuracy(train_pred, train_y_tf)
	test_accuracy = compute_accuracy(test_pred, test_y_tf)

with tf.Session(graph=graph) as sess:
	tf.initialize_all_variables().run()
	print("Initialized")
	for step in range(2000):
		offset = (step*batch_size)%(train_X.shape[0]-batch_size)
		batch_X = train_X[offset : (offset+batch_size), :]
		batch_y = train_y[offset : (offset+batch_size), :]
		feed_dict = {train_X_tf : batch_X, train_y_tf : batch_y}
		_, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy], feed_dict=feed_dict)
		vacc = valid_accuracy.eval()
		print("Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%"\
			%(step, tloss, (tacc*100), (vacc*100)))
	print("Done training")
	test_acc = test_accuracy.eval()
	print("test accuracy: %.2f%%" %((test_acc)*100))


