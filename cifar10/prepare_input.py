import numpy as np
from six.moves import cPickle as pickle
from sklearn.model_selection import train_test_split
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/")
from preprocess import *

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

def load_data(data_dir):
    # Load training and testing data
    train_fnroot = data_dir+'data_batch_'
    test_filename = data_dir+'test_batch'
    train_dataset = None
    test_dataset = None
    print("Loading the training data...")
    for i in range(1,6):
        train_filename = train_fnroot + str(i)
        batch = unpickle(train_filename)
        if i==1:
            train_dataset = batch['data']
            train_labels = np.array(batch['labels'])
        else:
            train_dataset = np.concatenate((train_dataset, batch['data']), axis=0)
            train_labels = np.concatenate((train_labels, batch['labels']))
    print("Loading the testing data...")
    test_batch = unpickle(test_filename)
    test_dataset = test_batch['data']
    test_labels = np.array(test_batch['labels'])
    return train_dataset, train_labels,  test_dataset, test_labels

def augment_data_cifar10(features, labels):
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

def preprocess_data_cifar10(X, y, num_labels):
    # 1) One-hot encode labels
    # 2) Random Permute samples
    # 3) Change datatype to np.float32 to speed up
    # zero-mean and zca-whitening
    y_encoded = np.arange(num_labels)==y[:, None]
    perm = np.random.permutation(y_encoded.shape[0])
    X_preprocessed = X[perm]
    y_encoded = y_encoded[perm]
    return X_preprocessed.astype(np.float32), y_encoded.astype(np.float32)

def whiten_data(X):
    X_centered = center_data(X)
    X_whitened = zca_whiten(X_centered)
    return X_whitened

def one_hot_encode(y, num_labels):
    '''One-Hot Encode labels'''
    y_encoded = np.arange(num_labels)==y[:, None]
    return y_encoded

def prepare_cifar10_input_new(data_dir, augmentation=True):
    print("Load data", "*"*32)
    train_dataset, train_labels, test_dataset, test_labels = load_data(data_dir)

    # Split 20% of training set as validation set
    print("Split training and validation set", "."*32)
    train_dataset, valid_dataset, train_labels, valid_labels = \
    train_test_split(train_dataset, train_labels, test_size=5000,\
    random_state=897, stratify=train_labels)

    # Reshape the data into pixel by pixel by RGB channels
    print("Reformat data", "."*32)
    train_dataset = np.rollaxis(train_dataset.reshape((-1,3,32,32)), 1, 4)
    valid_dataset = np.rollaxis(valid_dataset.reshape((-1,3,32,32)), 1, 4)
    test_dataset = np.rollaxis(test_dataset.reshape((-1,3,32,32)), 1, 4)
    print('Dataset\t\tFeatureShape\t\tLabelShape')
    print('Training set:\t', train_dataset.shape,'\t', train_labels.shape)
    print('Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape)
    print('Testing set:\t', test_dataset.shape, '\t', test_labels.shape)

    if augmentation:
        print('Augmenting training data......')
        train_dataset, train_labels = augment_data(train_dataset, train_labels)
    
    print("Whitening data......")
    train_X = whiten_data(train_dataset.reshape((train_labels.shape[0],-1)))
    valid_X = whiten_data(valid_dataset.reshape((valid_labels.shape[0],-1)))
    test_X  = whiten_data(test_dataset.reshape((test_labels.shape[0],-1)))
    train_dataset = train_X.reshape((-1,32,32,3))
    valid_dataset = valid_X.reshape((-1,32,32,3))
    test_dataset = test_X.reshape((-1,32,32,3))
    print("One-Hot Encoding labels......")
    num_labels = 10
    train_labels = one_hot_encode(train_labels, num_labels)
    valid_labels = one_hot_encode(valid_labels, num_labels)
    test_labels = one_hot_encode(test_labels, num_labels)

    train_dataset = train_dataset.astype(np.float32)
    valid_dataset = valid_dataset.astype(np.float32)
    test_dataset = test_dataset.astype(np.float32)
    train_labels = train_labels.astype(np.float32)
    valid_labels = valid_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    dataset_list = [train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels]
    print('Dataset\t\tFeatureShape\t\tLabelShape')
    print('Training set:\t', train_dataset.shape,'\t', train_labels.shape)
    print('Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape)
    print('Testing set:\t', test_dataset.shape, '\t', test_labels.shape)
    return dataset_list



def prepare_cifar10_input(data_dir, augmentation=False):
    # Load Data
    print("Load data", "."*32)
    train_dataset, train_labels, test_dataset, test_labels = load_data(data_dir)
    train_dataset = whiten_data(train_dataset)
    test_dataset = whiten_data(test_dataset)

    # Split 20% of training set as validation set
    print("Split training and validation set", "."*32)
    train_dataset, valid_dataset, train_labels, valid_labels = \
    train_test_split(train_dataset, train_labels, test_size=5000,\
    random_state=897, stratify=train_labels)
    # Print out data shapes
    print('Dataset\t\tFeatureShape\tLabelShape')
    print('Training set:\t', train_dataset.shape,'\t', train_labels.shape)
    print('Validation set:\t', valid_dataset.shape,'\t', valid_labels.shape)
    print('Testing set:\t', test_dataset.shape, '\t', test_labels.shape)

    # Reshape the data into pixel by pixel by RGB channels
    print("Reformat data", "."*32)
    train_dataset = np.rollaxis(train_dataset.reshape((-1,3,32,32)), 1, 4)
    valid_dataset = np.rollaxis(valid_dataset.reshape((-1,3,32,32)), 1, 4)
    test_dataset = np.rollaxis(test_dataset.reshape((-1,3,32,32)), 1, 4)
    print('Dataset\t\tFeatureShape\t\tLabelShape')
    print('Training set:\t', train_dataset.shape,'\t', train_labels.shape)
    print('Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape)
    print('Testing set:\t', test_dataset.shape, '\t', test_labels.shape)

    # Dataset Parameters
    image_size = 32
    num_labels = 10
    num_channels = 3

    # Data Augmentation
    if augmentation:
        train_dataset, train_labels = augment_data_cifar10(train_dataset, train_labels)

    # Data Preprocess: change datatype; center the data
    print("Preprocess data", "."*32)
    train_dataset, train_labels = preprocess_data_cifar10(train_dataset, train_labels, num_labels)
    valid_dataset, valid_labels = preprocess_data_cifar10(valid_dataset, valid_labels, num_labels)
    test_dataset,  test_labels  = preprocess_data_cifar10(test_dataset,  test_labels,  num_labels)
    dataset_list = [train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels]
    print('Dataset\t\tFeatureShape\t\tLabelShape')
    print('Training set:\t', train_dataset.shape,'\t', train_labels.shape)
    print('Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape)
    print('Testing set:\t', test_dataset.shape, '\t', test_labels.shape)
    return dataset_list

def plot_data(data):
    epoch = range(data['train_losses'].shape[0])
    plt.subplot(211)
    plt.plot(epoch, data['train_losses'], label= 'training loss')
    plt.plot(epoch, data['valid_losses'], label='validation loss')
    plt.ylim([-.1,4])
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(epoch, data['train_acc'], label='training accuracy')
    plt.plot(epoch, data['valid_acc'], label='validation accuracy')
    plt.ylim([0.0, 1.1])
    plt.legend(loc='lower right')
    plt.show()

def train_data_summary(filename):
    data = unpickle(filename)
    plot_data(data)
    print("Final Valid Accuracy: %.4f \t Test Accuracy: %.4f" \
        %(data['valid_acc'][-1], data['test_acc']))

if __name__=='__main__':
    data_dir = "./data/"
    #prepare_cifar10_input(data_dir)
    prepare_cifar10_input_new(data_dir)