import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
from sklearn import preprocessing

def load_data(data_filename):
	data = pd.read_csv(data_filename)
	if data.shape[1]==785: # training set
		y = data[['label']]
		X = data[data.columns[1:]]
	else: # testing set
		y = None
		X = data
	X = X.as_matrix()
	X = preprocess_data(X)
	X = X.astype(np.float32)
	if y is not None:
		y = y.as_matrix().astype(np.float32)
		y = np.arange(10)==y[:, None]
		y = y.astype(np.float32)
		y = np.squeeze(y, axis=1)
	return X, y

def preprocess_data(X): #consider add whitening
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