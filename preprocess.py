"""
Provide functions for preprocessing image inputs into (convolutional) neural network.
center_data: de-mean data for each feature and produces zero-mean data
pca_whiten: PCA whitening de-meaned data
zca_whiten: ZCA whitening de-meaned data
Implementation reference: http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening

Data Augmentation Utilities:
random_rotate:
"""

import numpy as np
from scipy.linalg import eigh, svd

def center_data(X, axis=0):
	"""
	Remove the mean along given axis. 
	Default axis=0 to remove mean for each feature/col.
	Setting axis=1 to remove mean for each sample/row.
	Return a new copy of centered zero-mean array.
	"""
	avg = np.mean(X, axis)
	X_centered = X - avg
	return X_centered

def pca_whiten(X):
	"""
	Principle Component Analysis - Data Whitening 
	Perform PCA whitening on data matrix X
	X needs to be a centered/de-meaned 2D numpy array with shape of nxm.
	n - number of samples;
	m - number of features.
	Using scipy linalg libaray instead of numpy linalg library due to that 
	scipy wrapper is more complete than numpy.
	Using svd instead of eigh for svd's numerical stability.
	epsilon for ensuring numerical stability and prevent from zero devision.
	"""
	epsilon = 1e-8
	sigma = np.dot(X.T, X)/X.shape[0]
	u, s, v = svd(sigma)
	s_sqrt = 1.0/np.sqrt(s + epsilon)
	s_diag = np.diag(s_sqrt)
	whiten_factor = np.dot(s_diag, u.T)
	X_whitened = np.dot(X, whiten_factor)
	return X_whitened

def zca_whiten(X):
	"""
	Zero-phase Component Analysis - Data Whitenening - link to Independent Component Aanalysis
	Perform ZCA whitening on data matrix X
	X needs to be a centered/de-meaned 2D numpy array with shape of nxm.
	n - number of samples;
	m - number of features.
	Using scipy linalg libaray instead of numpy linalg library due to that 
	scipy wrapper is more complete than numpy.
	Using svd instead of eigh for svd's numerical stability.
	epsilon for ensuring numerical stability and prevent from zero devision.
	"""
	epsilon = 1e-8
	sigma = np.dot(X.T, X)/X.shape[0]
	u, s, v = svd(sigma)
	s_sqrt = 1.0/np.sqrt(s + epsilon)
	s_diag = np.diag(s_sqrt)
	whiten_factor = np.dot(np.dot(u, s_diag), u.T)
	X_whitened = np.dot(X, whiten_factor)
	return X_whitened

def random_flip180(X):
	"""
	Randomly (50%) flip the images 180 degree: up down flip and left right flip
	X is a 4D array with the 1st D denoting number of samples, 2nd and 3rd Ds
	denoting the spatial dimensions, and the 4th D denoting the channels.
	Returns the randomly flipped X.
	"""
	Xc = X.copy()
	ud_ind = np.random.binomial(1, 0.5, X.shape[0]).astype(np.bool)
	lf_ind = np.invert(ud_ind)
	Xc[ud_ind] = X[ud_ind][:, ::-1, :, :]
	Xc[lf_ind] = X[lf_ind][:, :, ::-1, :]
	return Xc

def adjust_contrast(X):
	"""
	Adjust the imput images' brightness contrast.
	X is a 4D array with the 1st D denoting number of samples, 2nd and 3rd Ds 
	denoting the spatial dimensions, and the 4th D denoting the channels.
	Returns a ndarray in the same shape as X with brightness contrast adjusted.
	"""
	Xc = X.copy()
	darken_ind = np.random.binomial(1, 0.5, X.shape[0]).astype(np.bool)
	brighten_ind = np.invert(darken_ind)
	Xc[darken_ind] = darken_image(X[darken_ind])
	Xc[brighten_ind] = brighten_image(X[brighten_ind])
	return Xc

def darken_image(X):
	'''Darken image color contrast. X - 3D or 4D ndarray'''
	phi = 0.99
	theta = 0.997
	darkened = (254./phi)*(X/(254./theta))**1.0
	return darkened

def brighten_image(X):
	'''Brighten image color contrast. X - 3D or 4D ndarray - uncertain?'''
	phi = .993
	theta = 1.0
	brightened = X*(254./phi)*(1./(255./theta))**2.
	return brightened

def augment_data(X, y):
	"""
	Double sample size by performaning data augmentation.
	(1) Randomly flipping image samples horizontally or vertically
	(2) Randomly adjust image color contrast
	(3) Concatenate augmented data and original data
	(4) Randomly shuffle data
	"""
	X_aug = random_flip180(X)
	X_aug = adjust_contrast(X_aug)
	y_aug = y.copy()
	X_all = np.concatenate((X, X_aug), axis=0)
	y_all = np.concatenate((y, y_aug), axis=0)
	perm = np.random.permutation(X_all.shape[0])
	X_perm = X_all[perm]
	y_perm = y_all[perm]
	return X_perm, y_perm



