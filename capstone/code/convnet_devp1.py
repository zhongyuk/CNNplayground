import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.model_selection import train_test_split
import time

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
    train_fnroot = '../../cifar10/data/data_batch_'
    test_filename = '../../cifar10/data/test_batch'
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
    # 2) Change datatype
    # 3) Change datatype to np.float32 to speed up
    avg = np.mean(X, 0)
    repeat_avg = np.broadcast_to(avg, X.shape)
    X_centered = X - repeat_avg
    y_encoded = np.arange(num_labels)==y[:, None]
    return X_centered.astype(np.float32), y_encoded.astype(np.float32)

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

def convnet_model_stack(data, weights, biases, dropout=True, keep_prob=.5):
    # Linearly Stacked CNN
    # Construct convolution layers
    conv = conv_layer(data, weights['conv1'], biases['conv1'])
    pool = pool_layer(conv, 'max')
    conv = conv_layer(pool, weights['conv2'], biases['conv2'], padding="SAME")
    pool = pool_layer(conv, 'max')
    conv = conv_layer(pool, weights['conv3'], biases['conv3'])
    pool = pool_layer(conv, 'max')
    # Reshape data from 4D into 2D, prepare for fully connected layers
    shape = pool.get_shape().as_list()
    data = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
    # Construct fully connected layers
    if dropout:
        fc = full_layer(data, weights['fc1'], biases['fc1'], True, keep_prob)
    #fc = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], True, keep_prob)
    else:
        fc = full_layer(data, weights['fc1'], biases['fc1'], False)
    #fc = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], False)
    output = full_layer(fc, weights['fc2'], biases['fc2'], False)
    return output

def convnet_model_inception(data, weights, biases, dropout=True, keep_prob=.5):
    # A Simple Inception CNN
    # Construct convolution layers
    conv_share = conv_layer(data, weights['conv1'], biases['conv1'])
    pool = pool_layer(conv_share, 'max')
    # Inception Layer
    conv_top = conv_layer(pool, weights['conv2'], biases['conv2'])
    pool_top = pool_layer(conv_top, 'max')
    conv_bot = conv_layer(pool, weights['conv3'], biases['conv3'])
    pool_bot = pool_layer(conv_bot, 'max')
    # Concatenate layer
    concat = tf.concat(3, [pool_top, pool_bot])
    pool = pool_layer(concat, 'avg')
    # Reshape data from 4D into 2D, prepare for fully connected layers
    shape = pool.get_shape().as_list()
    data = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
    # Construct fully connected layers
    if dropout:
        fc = full_layer(data, weights['fc1'], biases['fc1'], True, keep_prob)
    #fc = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], True, keep_prob)
    else:
        fc = full_layer(data, weights['fc1'], biases['fc1'], False)
    #fc = full_layer(fc, weights['fc2_wt'], biases['fc2_bi'], False)
    output = full_layer(fc, weights['fc2'], biases['fc2'], False)
    return output

def train_convnet(graph, model, tf_data, convnet_shapes, hyperparams, steps, minibatch=False, *args):
    # Default exponential decay learning rate and AdamOptimizer
    print "Prepare network parameters", "."*32
    with graph.as_default():
        # Setup training, validation, testing dataset
        tf_train_dataset = tf_data['train_X']
        tf_train_labels = tf_data['train_y']
        tf_valid_dataset = tf_data['valid_X']
        tf_valid_lables = tf_data['valid_y']
        tf_test_dataset = tf_data['test_X']
        tf_test_labels = tf_data['test_y']
        # Initialize weights and biases
        weights, biases = {}, {}
        for key in convnet_shapes.keys():
            weights[key], biases[key] = initialize_variable(convnet_shapes[key], \
                                        hyperparams['init_mean'], hyperparams['init_std'])
        # HyperParameters
        keep_prob = hyperparams['keep_prob']
        init_lr = hyperparams['init_lr']
        decay_steps = hyperparams['decay_steps']
        decay_rate = hyperparams['decay_rate']
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate)
        
        # Compute Loss Function
        train_logits = model(tf_train_dataset, weights, biases, True, keep_prob)
        train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
        # Optimizer
        tfoptimizer = hyperparams['optimizer']
        optimizer = tfoptimizer(learning_rate).minimize(train_loss, global_step=global_step)
        # Prediction
        train_prediction = tf.nn.softmax(train_logits)
        valid_logits = model(tf_valid_dataset, weights, biases, False)
        valid_prediction = tf.nn.softmax(valid_logits)
        valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, tf_valid_lables))
        if tf_test_dataset!=None:
            test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, biases, False))
        else:
            test_prediction = None
    
    # Train Convnet
    num_steps = steps
    train_losses = np.zeros(num_steps)
    valid_losses = np.zeros(num_steps)
    train_acc = np.zeros(num_steps)
    valid_acc = np.zeros(num_steps)
    
    print "Start training", '.'*32
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            #if ((step>0)):
            #print "Time cost for the previous epoch is: %.2f seconds" %(time.time()-t)
            t = time.time()
            # Handle MiniBatch
            if minibatch:
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset+batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset+batch_size), :]
                feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels}
            else:
                feed_dict = {}
            # Run session...
            _, tl, predictions = session.run([optimizer, train_loss, train_prediction], feed_dict=feed_dict)
            train_losses[step] = tl
            if minibatch:
                train_acc[step] = accuracy(predictions, batch_labels)
            else:
                train_acc[step] = accuracy(predictions, tf_train_labels.eval())
            # Compute validation set accuracy batch by batch
            valid_losses[step] = valid_loss.eval()
            valid_acc[step] = accuracy(valid_prediction.eval(), tf_valid_lables.eval())
            if ((step % 200 == 0)):
                print('Training loss at step %d: %f' % (step, tl))
                print('Training accuracy: %.1f%%' % (train_acc[step]*100))
                #print('Validation loss at step %d: %f' % (step, valid_losses[step]))
                print('Validation accuracy: %.1f%%' % (valid_acc[step]*100))
        print "Finished training", '.'*32
        # Compute test set accuracy
        if test_prediction!=None:
            test_acc = accuracy(test_prediction.eval(), tf_test_labels.eval())
            print("Test accuracy: %1.f%%" %(test_acc*100))
        else:
            test_acc = None
    # Group training data into a dictionary
    training_data = {'train_losses' : train_losses, 'train_acc' : train_acc, \
                     'valid_losses' : valid_losses, 'valid_acc' : valid_acc, 'test_acc' : test_acc}
    return graph, training_data



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

    # Augment Data
    print "Augment data", '.'*32
    train_dataset, train_labels = augment_data(train_dataset, train_labels)
    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape

    # Dataset Parameters
    image_size = 32
    num_labels = 10
    num_channels = 3
    
    # Data Preprocess: change datatype; center the data
    print "Preprocess data", "."*32
    train_dataset, train_labels = preprocess_data(train_dataset, train_labels, num_labels)
    valid_dataset, valid_labels = preprocess_data(valid_dataset, valid_labels, num_labels)
    test_dataset,  test_labels  = preprocess_data(test_dataset,  test_labels,  num_labels)
    print 'Dataset\t\tFeatureShape\t\tLabelShape'
    print 'Training set:\t', train_dataset.shape,'\t', train_labels.shape
    print 'Validation set:\t', valid_dataset.shape, '\t', valid_labels.shape
    print 'Testing set:\t', test_dataset.shape, '\t', test_labels.shape
    
    # Network parameters
    batch_size = 512
    kernel_size3 = 3
    kernel_size5 = 5
    num_filter = 32
    fc_size1 = 256

    # Setup shapes for each layer in the convnet
    convnet_shapes = {'conv1' : [kernel_size5, kernel_size5, num_channels, num_filter],
                      'conv2' : [kernel_size3, kernel_size3, num_filter, num_filter],
                      'conv3' : [kernel_size5, kernel_size5, num_filter, num_filter],
                      'fc1'   : [(image_size/2/2/2)**2*num_filter*2, fc_size1],
                      'fc2'   : [fc_size1, num_labels]}

    # Prepare data for tensorflow
    graph = tf.Graph()
    with graph.as_default():
        tf_data = {'train_X': tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)),
                   'train_y': tf.placeholder(tf.float32, shape=(batch_size, num_labels)),
                   'valid_X': tf.constant(valid_dataset), 'valid_y': tf.constant(valid_labels),
                   'test_X' : tf.constant(test_dataset),  'test_y' : tf.constant(test_labels)}
        tfoptimizer = tf.train.GradientDecentOptimizer

    # HyperParameters
    hyperparams = {'keep_prob': 1.0, 'init_mean': 0.0, 'init_std': 0.008,
                   'init_lr': 0.00032, 'decay_rate': .5, 'decay_steps': 100,
                   'optimizer': tfoptimizer}

    # Setup computation graph and train convnet
    steps = 2001
    #model, save_data_name = convnet_model_stack, 'training_data_stack'
    model, save_data_name = convnet_model_inception, 'training_data_inception'
    _, training_data = train_convnet(graph, model, tf_data, convnet_shapes, hyperparams, steps, True, train_dataset,train_labels)

    # Save data
    with open(save_data_name, 'w') as fh:
        pickle.dump(training_data, fh)



