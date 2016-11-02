import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from sklearn.cross_validation import train_test_split
import time
from tensorflow.python.training import moving_averages
from prepare_input import *

def _initialize_wts_bis(variable_scope, shape, initializer=tf.truncated_normal_initializer(stddev=.01)):
    with tf.variable_scope(variable_scope) as scope:
        w = tf.get_variable("wt", shape, initializer=initializer)
        b = tf.get_variable("bi", shape[-1], initializer=tf.constant_initializer(1.0))
        variable_summaries(w, variable_scope+'/weights')
        variable_summaries(b, variable_scope+'/biases')
        scope.reuse_variables()

def _initialize_BN(variable_scope, depth):
    with tf.variable_scope(variable_scope+'/BatchNorm') as scope:
        gamma = tf.get_variable("gamma", depth, initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", depth, initializer=tf.constant_initializer(0.0))
        moving_avg = tf.get_variable("moving_avg", depth, initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", depth, initializer=tf.constant_initializer(1.0), trainable=False)
        #variable_summaries(gamma, variable_scope+'/gamma')
        #variable_summaries(beta, variable_scope+'/beta')
        #variable_summaries(moving_avg, variable_scope+'/moving_mean')
        #variable_summaries(moving_var, variable_scope+'/moving_std')
        scope.reuse_variables()
        
def initialize_variables(cnn_shapes, initializer, batch_norm=False):
    for item in cnn_shapes:
        variable_scope, shape = item[0], item[1]
        _initialize_wts_bis(variable_scope, shape, initializer)
        if batch_norm:
            _initialize_BN(variable_scope, shape[-1])

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        avg = tf.reduce_mean(var)
        tf.scalar_summary('mean/'+name, avg)
        with tf.name_scope('std'):
            std = tf.sqrt(tf.reduce_mean(tf.square(var - avg)))
        tf.scalar_summary('std/'+name, std)
        tf.scalar_summary('max/'+name, tf.reduce_max(var))
        tf.scalar_summary('min/'+name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def pool_layer(x, method='max', kernel=2, stride=2, padding='SAME'):
    if method=='max':
        return tf.nn.max_pool(x, [1,kernel,kernel,1], [1,stride,stride,1], padding=padding)
    elif method=='avg':
        return tf.nn.avg_pool(x, [1,kernel,kernel,1], [1,stride,stride,1], padding=padding)
    else:
        raise ValueError(method+' pooling method does not exist')

def conv_layer(x, variable_scope, stride=1, padding='SAME'):
    with tf.variable_scope(variable_scope, reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        y = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding) + b
        variable_summaries(w, variable_scope+'/weights')
        variable_summaries(b, variable_scope+'/biases')
        tf.histogram_summary(variable_scope+'/preBN', y)
    return y

def fc_layer(x, variable_scope):
    with tf.variable_scope(variable_scope, reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        y = tf.matmul(x, w) + b
        variable_summaries(w, variable_scope+'/weights')
        variable_summaries(b, variable_scope+'/biases')
        tf.histogram_summary(variable_scope+'/preBN', y)
    return y

def batch_norm_layer(x, variable_scope, is_training, epsilon=0.001, decay=.999):
    with tf.variable_scope(variable_scope+'/BatchNorm', reuse=True):
        gamma, beta = tf.get_variable("gamma"), tf.get_variable("beta")
        moving_avg, moving_var = tf.get_variable("moving_avg"), tf.get_variable("moving_var")
        variable_summaries(gamma, variable_scope+'/gamma')
        variable_summaries(beta, variable_scope+'/beta')
        variable_summaries(moving_avg, variable_scope+'/moving_mean')
        variable_summaries(moving_var, variable_scope+'/moving_std')
        shape = x.get_shape().as_list()
        control_inputs = []
        if is_training:
            avg, var = tf.nn.moments(x, range(len(shape)-1))
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
        with tf.control_dependencies(control_inputs):
            y = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
        tf.histogram_summary(variable_scope+'/postBN', y)
    return y

def cnn_model(data, variable_scopes, is_training, batch_norm, keep_prob):
    x = data
    # 3 repetitive conv structures: conv -> batch norm -> ReLu -> max pool
    for var_scope in variable_scopes[:3]:
        x = conv_layer(x ,var_scope)
        if batch_norm:
            x = batch_norm_layer(x, var_scope, is_training)
        tf.histogram_summary(var_scope+'/preActivation', x)
        x = tf.nn.relu(x)
        tf.histogram_summary(var_scope+'/postActivation', x)
        x = pool_layer(x)
    # reshape x to prepare for fully connected layer
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    # fully connected layer: fc1 -> batch norm -> ReLu -> dropout
    x = fc_layer(x, variable_scopes[3])
    if batch_norm:
        x = batch_norm_layer(x, variable_scopes[3], is_training)
    tf.histogram_summary(var_scope+'/preActivation', x)
    x = tf.nn.relu(x)
    tf.histogram_summary(var_scope+'/postActivation', x)
    if is_training:
        x = tf.nn.dropout(x, keep_prob)
    # fully connected layer: fc2 -> batch norm -> ReLu -> output
    x = fc_layer(x, variable_scopes[4])
    if batch_norm:
        x = batch_norm_layer(x, variable_scopes[4], is_training)
    tf.histogram_summary(var_scope+'/preActivation', x)
    y = tf.nn.relu(x)
    tf.histogram_summary(var_scope+'/postActivation', y)
    return y

def train_cnn(graph, model, tf_data, cnn_shapes, hyperparams, epoches, *args):
    print "Prepare network parameters", "."*32
    with graph.as_default():
        # Setup training, validation, testing dataset
        tf_train_dataset, tf_train_labels = tf_data['train_X'], tf_data['train_y']
        tf_valid_dataset, tf_valid_labels = tf_data['valid_X'], tf_data['valid_y']
        tf_test_dataset , tf_test_labels  = tf_data['test_X'] , tf_data['test_y']
        # Initialize Weights and Biases
        scopes = zip(*convnet_shapes)[0]
        batch_norm = hyperparams['batch_norm']
        initialize_variables(cnn_shapes, initializer=hyperparams['initializer'], batch_norm=batch_norm)
          
        # Unwrap HyperParameters
        l2_reg = hyperparams['l2_reg'] # regularization penality factor
        keep_prob, tfoptimizer = hyperparams['keep_prob'], hyperparams['optimizer']
        init_lr,  global_step = hyperparams['init_lr'], tf.Variable(0)
        decay_steps, decay_rate = hyperparams['decay_steps'], hyperparams['decay_rate']
        learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate, staircase=True)
          
        # Compute Loss Function and Predictions
        train_logits = model(tf_train_dataset, scopes, True, batch_norm, keep_prob)
        # Without regularization
        train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
        tf.scalar_summary('train loss', train_loss)
        # With L2 regularization applied to fully connected layers
        #l2_reg_loss = 0
        #with tf.variable_scope('fc1', reuse=True):
            #l2_reg_loss += tf.nn.l2_loss(tf.get_variable('wt')) + tf.nn.l2_loss(tf.get_variable('bi'))
        #with tf.variable_scope('fc2', reuse=True):
            #l2_reg_loss += tf.nn.l2_loss(tf.get_variable('wt')) + tf.nn.l2_loss(tf.get_variable('bi'))
        #train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels) + l2_reg*l2_reg_loss)
        train_prediction = tf.nn.softmax(train_logits)
        # Optimizer
        optimizer = tfoptimizer(learning_rate).minimize(train_loss, global_step=global_step)
          
        valid_logits = model(tf_valid_dataset, scopes, False, batch_norm, keep_prob)
        valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits,tf_valid_labels))
        tf.scalar_summary('validation loss', valid_loss)
        valid_prediction = tf.nn.softmax(valid_logits)
        test_prediction = tf.nn.softmax(model(tf_test_dataset, scopes, False, batch_norm, keep_prob))
        merge_summary_op = tf.merge_all_summaries()
        
  # Train Convnet
    train_losses, valid_losses = np.zeros(epoches), np.zeros(epoches)
    train_acc, valid_acc = np.zeros(epoches), np.zeros(epoches)
    
    
    print "Start training", '.'*32
    with tf.Session(graph=graph) as session:
        writer = tf.train.SummaryWriter('logs/', graph=session.graph)
        tf.initialize_all_variables().run()
        print('Initialized')
        for epoch in range(epoches):
            t = time.time()
            offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset+batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset+batch_size), :]
            feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels}
            # Run session...
            _, tl, predictions, summaries = session.run([optimizer, train_loss, train_prediction, merge_summary_op], feed_dict=feed_dict)
            writer.add_summary(summaries, epoch)
            
            train_losses[epoch] = tl
            train_acc[epoch] = accuracy(predictions, batch_labels)
            #tf.scalar_summary('train accuracy', train_acc[epoch])
            # Compute validation set accuracy
            valid_losses[epoch] = valid_loss.eval()
            valid_acc[epoch] = accuracy(valid_prediction.eval(), tf_valid_labels.eval())
            #tf.scalar_summary('validation accuracy', train_acc[epoch])
            print('Epoch: %d:\tLoss: %f\t\tTime cost: %1.f\t\tTrain Acc: %.2f%%\tValid Acc: %2.f%%\tLearning rate: %.6f' \
                %(epoch, tl, (time.time()-t), (train_acc[epoch]*100), (valid_acc[epoch]*100),learning_rate.eval(),))
        print "Finished training", '.'*32
        # Compute test set accuracy
        test_acc = accuracy(test_prediction.eval(), tf_test_labels.eval())
        print("Test accuracy: %2.f%%" %(test_acc*100))
        writer.close()
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
    batch_size = 128
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
    #graph = tf.Graph()
    #with graph.as_default():
    #tf_data = {'train_X': tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)),
    #'train_y': tf.placeholder(tf.float32, shape=(batch_size, num_labels)),
    #'valid_X': tf.constant(valid_dataset), 'valid_y': tf.constant(valid_labels),
    #'test_X' : tf.constant(test_dataset),  'test_y' : tf.constant(test_labels)}
    #tfoptimizer = tf.train.AdamOptimizer
      
    # HyperParameters
    hyperparams = {'keep_prob': 0.47, 'init_lr': 0.002, 'decay_rate': .9, 'decay_steps': 100,
                   'optimizer': tfoptimizer, 'l2_reg': 0.096, 'batch_norm': True,
                    'initializer': tf.truncated_normal_initializer(stddev=.015)} #tf.contrib.layers.variance_scaling_initializer()}
      
    # Setup computation graph and train convnet
    #steps = 31
    #model, save_data_name = cnn_model, 'training_data_stack3.1'
    #model, save_data_name = convnet_inception, 'training_data_inception'
    #_, training_data = train_cnn(graph, model, tf_data, convnet_shapes, hyperparams, \
                                    steps, True, train_dataset, train_labels, batch_size)
      
    # Save data
    #with open(save_data_name, 'w') as fh:
        #pickle.dump(training_data, fh)
    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        

                          
                          





                                     