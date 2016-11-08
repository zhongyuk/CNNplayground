import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.cross_validation import train_test_split

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

def load_data():
    # Load training and testing data
    train_fnroot = 'cifar-10-batches-py/data_batch_'
    test_filename = 'cifar-10-batches-py/test_batch'
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
  
def augment_data():
    # 1) Mirror Reflection
    # 2) Random Corp
    # 3) Color Jitter
    pass

def initialize_variable(shape, mean=0.0, std=.1):
    # Initialize weights and biases based on given shape
    wt = tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=std ))
    bi = tf.Variable(tf.random_normal(shape=[shape[-1]], mean=mean, stddev=std))
    return wt, bi

def conv_layer(x, w, b, stride=1, padding='SAME'):
    # Perform a convolution layer computation followed by a ReLu activation
    # padding: "SAME" or "VALID" 
    conv = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding)
    relu = tf.nn.relu(conv + b)
    return relu

def pool_layer(x, method='max', kernel=2, stride=2, padding='SAME'):
    # Perform a down sampling layer computation - "max" : max pooling, "avg" : avg pooling
    if method=="max":
	return tf.nn.max_pool(x, [1,kernel,kernel,1], [1,stride,stride,1], padding=padding)
    elif method=='avg':
	return tf.nn.avg_pool(x, [1,kernel,kernel,1], [1,stride,stride,1], padding=padding)
    else:
	raise ValueError

def full_layer(x, w, b, dropout=True, keep_prob=.5):
    # Perform a fully connected layer computation followed by a ReLu activation
    # If dropout is True, drop out is performed
    fc = tf.nn.relu(tf.matmul(x,w) + b)
    if dropout:
	fc = tf.nn.dropout(fc, keep_prob)
    return fc

def convnet_model(data, weights, baises, dropout=True, keep_prob=.5):
    # Construct convolution layers
    conv = conv_layer(data, weights['conv1_wt'], biases['conv1_bi'])
    pool = pool_layer(conv, 'max')
    conv = conv_layer(pool, weights['conv2_wt'], biases['conv2_bi'], padding="SAME")
    pool = pool_layer(conv, 'max')
    conv = conv_layer(pool, weights['conv3_wt'], biases['conv3_bi'])
    pool = pool_layer(conv, 'max')
    # Reshape data from 4D into 2D, prepare for fully connected layers
    shape = pool.get_shape().as_list()
    data = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
    # Construct fully connected layers
    if dropout:
        fc = full_layer(data, weights['fc1_wt'], biases['fc1_bi'], True, keep_prob)
    #fc = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], True, keep_prob)
    else:
        fc = full_layer(data, weights['fc1_wt'], biases['fc1_bi'], False)
#fc = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], False)
    output = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], False)
    return output

 

if __name__=='__main__': 
    # Load Data
    print "Load data", "."*32
    train_dataset, train_labels, test_dataset, test_labels = load_data()

    # Split 20% of training set as validation set
    print "Split training and validation set", "."*32    
    train_dataset, valid_dataset, train_labels, valid_labels = \
    train_test_split(train_dataset, train_labels, test_size=10000,\
    random_state=897, stratify=train_labels)

    # Print out data shapes
    print 'Dataset\t\tFeatureShape\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape,'\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape

    # Reshape the data into pixel by pixel by RGB channels
    print "Reformat data", "."*32
    train_dataset = np.rollaxis(train_dataset.reshape((-1,3,32,32)), 1, 4)
    valid_dataset = np.rollaxis(valid_dataset.reshape((-1,3,32,32)), 1, 4)
    test_dataset = np.rollaxis(test_dataset.reshape((-1,3,32,32)), 1, 4)
    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape

    # Mirror Reflection
    #train_LRF = train_reshape[:,:,::-1,:]

    # Dataset Parameters
    image_size = 32
    num_labels = 10
    num_channels = 3

    # Data Preprocess
    print "Preprocess data", "."*32
    train_dataset = train_dataset.astype(np.float32)
    train_labels = (np.arange(num_labels)==train_labels[:,None]).astype(np.float32)

    valid_dataset = valid_dataset.astype(np.float32)
    valid_labels = (np.arange(num_labels)==valid_labels[:, None]).astype(np.float32)

    test_dataset = test_dataset.astype(np.float32)
    test_labels = (np.arange(num_labels)==test_labels[:,None]).astype(np.float32)
    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape

    # Network parameters
    batch_size = 128
    kernel_size3 = 3
    kernel_size5 = 5
    num_filter = 16
    fc_size1 = 256
    #fc_size2 = 64
    valid_batch_size, test_batch_size = 500, 500

    # ConvNet
    print "Prepare network parameters", "."*32
    graph = tf.Graph()

    with graph.as_default():
	# Setup Input
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.placeholder(tf.float32, shape=(valid_batch_size, image_size, image_size, num_channels))
        tf_test_dataset = tf.placeholder(tf.float32, shape=(test_batch_size, image_size, image_size, num_channels))
        
        # Setup Variables
        convnet_shapes = {'conv1' : [kernel_size5, kernel_size5, num_channels, num_filter],
			  'conv2' : [kernel_size3, kernel_size3, num_filter, num_filter],
              'conv3' : [kernel_size5, kernel_size5, num_filter, num_filter],
			  'fc1'   : [(image_size/2/2/2)**2*num_filter, fc_size1],
              'fc2'   : [fc_size1, num_labels]}
              #'fc3'   : [fc_size2, num_labels]}

        weights, biases = {}, {}
        weights['conv1_wt'], biases['conv1_bi'] = initialize_variable(convnet_shapes['conv1'])
        weights['conv2_wt'], biases['conv2_bi'] = initialize_variable(convnet_shapes['conv2'])
        weights['conv3_wt'], biases['conv3_bi'] = initialize_variable(convnet_shapes['conv3'])
        weights['fc1_wt'],   biases['fc1_bi']   = initialize_variable(convnet_shapes['fc1'])
        weights['fc2_wt'],   biases['fc2_bi']   = initialize_variable(convnet_shapes['fc2'])
        #weights['fc3_wt'],   biases['fc3_bi']   = initialize_variable(convnet_shapes['fc3'])

        # HyperParameters
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(.0001, global_step, 10000, .7, staircase=True)
        
        # Compute Loss Function
        logits = convnet_model(tf_train_dataset, weights, biases, True, keep_prob)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        
        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
        # Prediction
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(convnet_model(tf_valid_dataset, weights, biases, False))
        test_prediction = tf.nn.softmax(convnet_model(tf_test_dataset, weights, biases, False))

    # Setup training steps
    print "Start training", '.'*32
    num_steps = 30001
    loss_val = np.zeros(num_steps)
    train_acc = np.zeros(num_steps)
    valid_acc = np.zeros(num_steps)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : .7}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            
            loss_val[step] = l
            train_acc[step] = accuracy(predictions, batch_labels)

 	    # Compute validation set accuracy batch by batch
            i, valid_correct = 0, 0
            while i < valid_dataset.shape[0]:
                valid_data_batch = valid_dataset[i: i+valid_batch_size, :, :, :]
                valid_label_batch = valid_labels[i: i+valid_batch_size, :]
                valid_feed_dict = {tf_valid_dataset : valid_data_batch}
                valid_correct += count_correct(valid_prediction.eval(feed_dict=valid_feed_dict), valid_label_batch)
                i += valid_batch_size
            valid_acc[step] = float(valid_correct)/valid_dataset.shape[0]
            if ((step % 50 == 0) or (step<20)):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % (train_acc[step]*100))
                print('Validation accuracy: %.1f%%' % (valid_acc[step]*100))
	
	# Compute test set accuracy batch by batch
	print "Finished training", '.'*32
	i, test_correct = 0, 0
	while i < test_dataset.shape[0]:
	    test_data_batch = test_dataset[i: i+test_batch_size, :, :, :]
	    test_label_batch = test_labels[i: i+test_batch_size, :]
	    test_feed_dict = {tf_test_dataset : test_data_batch}
	    test_correct += count_correct(test_prediction.eval(feed_dict=test_feed_dict), test_label_batch)
	    i += test_batch_size
	test_acc = float(test_correct)/test_dataset.shape[0]
    print('Test accuracy: %.1f%%' % (test_acc*100))
    
    # Save training data
    training_data = {'loss_val' : loss_val, 'train_acc' : train_acc, 'valid_acc' : valid_acc, 'test_acc' : test_acc}
    with open('training_data', 'w') as fh:
	pickle.dump(training_data, fh)
