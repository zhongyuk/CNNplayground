import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../../")


def unpickle(file):
    # Load pickled data
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def accuracy(pred, labels):
    # Compute accuracy
    return (1.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1))/ pred.shape[0])

def count_correct(pred, labels):
    # Count number of correctly classified samples
    return np.sum(np.argmax(pred, 1) == np.argmax(labels, 1))

def generate_batch(features, labels, batch_size):
    # Generate a random small batch of data
    start = np.random.randint(0, features.shape[0]-batch_size)
    feature_batch, label_batch = features[start:start+batch_size,:,:,:], labels[start:start+batch_size,:]
    return feature_batch, label_batch

def load_data():
    # Load training and testing data
    train_fnroot = 'cifar10_data/data_batch_'
    test_filename = 'cifar10_data/test_batch'
    train_dataset = None
    test_dataset = None
    print "Loading the training data..."
    for i in range(1,6):
        train_filename = train_fnroot + str(i)
        batch = unpickle(train_filename)
        if i==1:
            train_dataset = batch['data']
            train_labels = np.array(batch['labels'])
        else:
            train_dataset = np.concatenate((train_dataset, batch['data']), axis=0)
            train_labels = np.concatenate((train_labels, batch['labels']))
    print "Loading the testing data..."
    test_batch = unpickle(test_filename)
    test_dataset = test_batch['data']
    test_labels = np.array(test_batch['labels'])
    return train_dataset, train_labels,  test_dataset, test_labels

def augment_data(features, labels):
    # 50% Upside Down Filp; 50% Mirror Flip
    ud_ind = np.random.binomial(1, .5, features.shape[0]).astype(np.bool)
    lf_ind = np.invert(ud_ind)
    ud_features, ud_labels = features[ud_ind, :,:,:], labels[ud_ind]
    ud_features = ud_features[:, ::-1, :, :]
    lf_features, lf_labels = features[lf_ind, :,:,:], labels[lf_ind]
    lf_features = lf_features[:, :, ::-1, :]
    cat_features = np.concatenate((features, ud_features, lf_features), axis=0)
    cat_labels = np.concatenate((labels, ud_labels, lf_labels))
    return cat_features, cat_labels

def preprocess_data(X, y, num_labels):
    # 1) Center the training data/Subtract Mean
    # 2) One-hot encode labels
    # 3) Random Permute samples
    # 4) Change datatype to np.float32 to speed up
    avg = np.mean(X, 0)
    repeat_avg = np.broadcast_to(avg, X.shape)
    X_centered = X - repeat_avg
    y_encoded = np.arange(num_labels)==y[:, None]
    perm = np.random.permutation(y_encoded.shape[0])
    X_centered = X_centered[perm]
    y_encoded = y_encoded[perm]
    return X_centered.astype(np.float32), y_encoded.astype(np.float32)
