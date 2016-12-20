import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
from sklearn import preprocessing
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/")
from preprocess import *
import matplotlib.pyplot as plt

def load_data(data_filename):
	data = pd.read_csv(data_filename)
	if data.shape[1]==785: # training set
		y = data[['label']]
		X = data[data.columns[1:]]
	else: # testing set
		y = None
		X = data
	X = X.as_matrix()
	# zero-mean and zca-whitening
	X_centered = center_data(X)
	X_whitened = zca_whiten(X_centered)
	#X_whitened = preprocess_data(X)
	X_preprocessed = X_whitened.astype(np.float32)
	if y is not None:
		y = y.as_matrix().astype(np.float32)
		y = np.arange(10)==y[:, None]
		y = y.astype(np.float32)
		y = np.squeeze(y, axis=1)
	return X_preprocessed, y

def preprocess_data(X):
	'''A simple preprocess func for zero-mean and unity-variance transformation'''
	X_scaled = preprocessing.scale(X)
	return X_scaled

def reshape_data(X):
	return np.reshape(X, [-1, 28, 28, 1])

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

def make_submission(pred, filename):
	pred_dict = {'ImageId' : range(1, pred.shape[0]+1),
				 'Label'   : pred}
	pred_df = pd.DataFrame(pred_dict)
	pred_df.to_csv(filename, index=False)

def unpickle(file):
    # Load pickled data
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def plot_sample(data_filename, sample_id):
	'''A function for plotting grayscale image of MNIST samples'''
	data = pd.read_csv(train_filename)
	X = data[data.columns[1:]]
	X = X.as_matrix()
	if sample_id<0 or sample_id>X.shape[0]-1:
		raise IndexError("Index Out of Bound!")
	sample = X[sample_id, :]
	sample_array = sample.reshape((28,28))
	plt.imshow(sample_array, cmap='gray')
	plt.show()