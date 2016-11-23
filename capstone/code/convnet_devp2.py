import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.model_selection import train_test_split
import time
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/cifar10")
from prepare_input import *

def initialize_variables(convnet_shapes, initializer=tf.truncated_normal_initializer(stddev=.01)):
    for item in convnet_shapes:
        scope_name, shape = item[0], item[1]
        with tf.variable_scope(scope_name) as scope:
            w = tf.get_variable("wt", shape, initializer = initializer)
            b = tf.get_variable("bi", shape[-1], initializer = tf.constant_initializer(1.0))
            scope.reuse_variables()

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

def full_layer(x, w, b):
    # Perform a fully connected layer computation followed by a ReLu activation
    # If dropout is True, drop out is performed
    fc = tf.nn.relu(tf.matmul(x,w) + b)
    return fc

def convnet_stack(data, scopes, dropout=True, keep_prob=.5):
    # Linearly Stacked CNN
    x = data
    for scope in scopes[:-1]:
        if scope[:-1]=='conv':
            with tf.variable_scope(scope, reuse=True):
                w = tf.get_variable("wt")
                b = tf.get_variable("bi")
                x = conv_layer(x, w, b)
            x = pool_layer(x, "max")
        else:
            with tf.variable_scope(scope, reuse=True):
                w = tf.get_variable("wt")
                b = tf.get_variable("bi")
                shape = w.get_shape().as_list()
                x = tf.reshape(x, [-1, shape[0]])
                x = full_layer(x, w, b)
                if dropout:
                    x = tf.nn.dropout(x, keep_prob)
    scope = scopes[-1]
    with tf.variable_scope(scope, reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        output = full_layer(x, w, b)
    return output


def convnet_inception(data, scopes, dropout=True, keep_prob=.5):
    # A Simple Inception CNN
    with tf.variable_scope(scopes[0], reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        x = conv_layer(data, w, b)
    x = pool_layer(x, "max")
    with tf.variable_scope(scopes[1], reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        x_top = conv_layer(x, w, b)
    pool_top = pool_layer(x_top, "max")
    with tf.variable_scope(scopes[2], reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        x_bot = conv_layer(x, w, b)
    pool_bot = pool_layer(x_bot, "max")
    # Concatenate layer
    concat = tf.concat(3, [pool_top, pool_bot])
    pool = pool_layer(concat, "avg")
    # Fully connected layer
    with tf.variable_scope(scopes[3], reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("b")
        shape = w.get_shape().as_list()
        x = tf.reshape(pool, [-1, shape[0]])
        x = full_layer(x, w, b)
        if dropout:
            x = tf.nn.dropout(x, keep_prob)
    with tf.variable_scope(scopes[4], reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        output = full_layer(x, w, b)
    return output


def train_convnet(graph, model, tf_data, convnet_shapes, hyperparams, epoches, minibatch=False, *args):
    # Default exponential decay learning rate and AdamOptimizer
    print "Prepare network parameters", "."*32
    with graph.as_default():
        # Setup training, validation, testing dataset
        tf_train_dataset, tf_train_labels = tf_data['train_X'], tf_data['train_y']
        tf_valid_dataset, tf_valid_labels = tf_data['valid_X'], tf_data['valid_y']
        tf_test_dataset , tf_test_labels  = tf_data['test_X'] , tf_data['test_y']
        # Initialize Weights and Biases
        scopes = zip(*convnet_shapes)[0]
        initialize_variables(convnet_shapes, initializer=hyperparams['initializer'])

        # Unwrap HyperParameters
        keep_prob, tfoptimizer = hyperparams['keep_prob'], hyperparams['optimizer']
        init_lr,  global_step = hyperparams['init_lr'], tf.Variable(0)
        decay_steps, decay_rate = hyperparams['decay_steps'], hyperparams['decay_rate']
        learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate, staircase=False)
        
        # Compute Loss Function and Predictions
        train_logits = model(tf_train_dataset, scopes, True, keep_prob)
        train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
        train_prediction = tf.nn.softmax(train_logits)
        # Optimizer
        optimizer = tfoptimizer(learning_rate).minimize(train_loss, global_step=global_step)
        
        valid_logits = model(tf_valid_dataset, scopes, False)
        valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, tf_valid_labels))
        valid_prediction = tf.nn.softmax(valid_logits)
        if tf_test_dataset!=None:
            test_prediction = tf.nn.softmax(model(tf_test_dataset, scopes, False))
        else:
            test_prediction = None
    
    # Train Convnet
    num_steps = epoches
    train_losses, valid_losses = np.zeros(num_steps), np.zeros(num_steps)
    train_acc, valid_acc = np.zeros(num_steps), np.zeros(num_steps)
    
    print "Start training", '.'*32
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
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
            # Compute validation set accuracy
            valid_losses[step] = valid_loss.eval()
            valid_acc[step] = accuracy(valid_prediction.eval(), tf_valid_labels.eval())
            if ((step % 200 == 0)):
                print('Epoch: %d:\t Loss: %f\t Time cost: %1.f\tTrain Acc: %.2f%%\tValid Acc: %2.f%%\tLearning rate: %.6f/' \
                      %(step, tl, (time.time()-t), (train_acc[step]*100), (valid_acc[step]*100),learning_rate.eval(),))
        print "Finished training", '.'*32
        # Compute test set accuracy
        if test_prediction!=None:
            test_acc = accuracy(test_prediction.eval(), tf_test_labels.eval())
            print("Test accuracy: %2.f%%" %(test_acc*100))
        else:
            test_acc = None
    # Group training data into a dictionary
    training_data = {'train_losses' : train_losses, 'train_acc' : train_acc, \
                     'valid_losses' : valid_losses, 'valid_acc' : valid_acc, 'test_acc' : test_acc}
    return graph, training_data



if __name__=='__main__':
    # Prepare cifar10 data input
    data_dir = "../../cifar10/data/"
    dataset_list = prepare_cifar10_input(data_dir)
    train_dataset, train_labels = dataset_list[0], dataset_list[1]
    valid_dataset, valid_labels = dataset_list[2], dataset_list[3]
    test_dataset , test_labels  = dataset_list[4], dataset_list[5]

    # Dataset Parameters
    image_size = 32
    num_labels = 10
    num_channels = 3
    
    # Network parameters
    batch_size = 512
    kernel_size3 = 3
    kernel_size5 = 5
    num_filter = 64
    fc_size1 = 512

    # Setup shapes for each layer in the convnet
    convnet_shapes = [['conv1', [kernel_size5, kernel_size5, num_channels, num_filter]],
                      ['conv2', [kernel_size3, kernel_size3, num_filter, num_filter]]  ,
                      ['conv3', [kernel_size5, kernel_size5, num_filter, num_filter]]  ,
                      ['fc1'  , [(image_size/2/2/2)**2*num_filter, fc_size1]]        ,
                      ['fc2'  , [fc_size1, num_labels]]]

    # Prepare data for tensorflow
    graph = tf.Graph()
    with graph.as_default():
        tf_data = {'train_X': tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)),
                   'train_y': tf.placeholder(tf.float32, shape=(batch_size, num_labels)),
                   'valid_X': tf.constant(valid_dataset), 'valid_y': tf.constant(valid_labels),
                   'test_X' : tf.constant(test_dataset),  'test_y' : tf.constant(test_labels)}
        tfoptimizer = tf.train.AdamOptimizer

    # HyperParameters
    hyperparams = {'keep_prob': 0.5, 'init_lr': 0.0007, 'decay_rate': .9, 'decay_steps': 100, 'optimizer': tfoptimizer,
        'initializer': tf.truncated_normal_initializer(stddev=.013)}#tf.contrib.layers.variance_scaling_initializer()}#

    # Setup computation graph and train convnet
    steps = 2501
    model, save_data_name = convnet_stack, 'training_data_stack2.6'
    #model, save_data_name = convnet_inception, 'training_data_inception'
    _, training_data = train_convnet(graph, model, tf_data, convnet_shapes, hyperparams, steps, True, train_dataset,train_labels)

    # Save data
    with open(save_data_name, 'w') as fh:
        pickle.dump(training_data, fh)



