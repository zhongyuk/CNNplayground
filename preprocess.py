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
    Default axis=0 to remove mean for each feature.
    Setting axis=1 to remove mean for each sample.
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



