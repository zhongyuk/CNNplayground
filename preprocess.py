"""
Provide functions for preprocessing image inputs into (convolutional) neural network.
center_data: de-mean data for each feature and produces zero-mean data
pca_whiten: PCA whitening de-meaned data
zca_whiten: ZCA whitening de-meaned data

Data Augmentation Utilities:
random_rotate:
"""

import numpy as np
from scipy.linalg import eigh, svd

def center_data(X, axis=0):
	"""
	Remove the mean along given axis. 
	Default axis=0 to remove mean for each feature.
	Setting axis=1 to remove mean for each sample.
	Return a new copy of centered zero-mean array.
	"""
	avg = np.mean(X, axis)
	X_centered = X - avg
	return X_centered

def pca_whiten(X):
	"""
	Perform PCA whitening on data matrix X
	X needs to be a centered/de-meaned 2D numpy array with shape of nxm.
	n - number of samples;
	m - number of features.
	Using scipy linalg libaray instead of numpy linalg library due to that 
	scipy wrapper is more complete than numpy.
	"""
	epsilon = 1e-15
	Xcov = np.dot(X.T, X)
	eigVal, eigVec = eigh(Xcov)
	eigVal_diag = np.diag(eigVal+epsilon)
	whiten_factor = np.sqrt(eigVal_diag)
	whiten_factor = np.dot(whiten_factor, eigVec.T)
	X_whitened = np.dot(X, whiten_factor)
	return X_whitened

def zca_whiten(X):
	"""
	Perform PCA whitening on data matrix X
	X needs to be a centered/de-meaned 2D numpy array with shape of nxm.
	n - number of samples;
	m - number of features.
	Using scipy linalg libaray instead of numpy linalg library due to that 
	scipy wrapper is more complete than numpy.
	"""
	epsilon = 1e-15
	Xcov = np.dot(X.T, X)
	eigVal, eigVec = eigh(Xcov)
	eigVal_diag = np.diag(eigVal+epsilon)
	eigVal_sqrt = np.sqrt(eigVal_diag)
	whiten_factor = np.dot(np.dot(eigVec, eigVal_sqrt), eigVec.T)
	X_whitened = np.dot(X, whiten_factor)
	return X_whitened



