import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
from six.moves import cPickle as pickle
import time
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/cifar10/")
from prepare_input import *

def initialize_variables(convnet_shapes, initializer=tf.truncated_normal_initializer(stddev=.01), batch_norm=False):
    for item in convnet_shapes:
        scope_name, shape = item[0], item[1]
        with tf.variable_scope(scope_name) as scope:
            w = tf.get_variable("wt", shape, initializer=initializer)
            b = tf.get_variable("bi", shape[-1], initializer=tf.constant_initializer(1.0))
            if batch_norm:
                with tf.variable_scope('BatchNorm') as bnscope:
                    gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
                    beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0))
                    moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
                    moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
                    bnscope.reuse_variables()
            scope.reuse_variables()


def conv_layer(x, scope, stride=1, padding='SAME'):
    # Perform a convolution layer computation
    # padding: "SAME" or "VALID"
    with tf.variable_scope(scope, reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        conv = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding=padding)
    return conv+b


def BatchNorm_layer(x, scope, train, epsilon=0.001, decay=.9):
    # Perform a batch normalization after a conv layer or a fc layer
    # gamma: a scale factor
    # beta: an offset
    # epsilon: the variance epsilon - a small float number to avoid dividing by 0
    with tf.variable_scope(scope, reuse=True):
        with tf.variable_scope('BatchNorm', reuse=True) as bnscope:
            gamma, beta = tf.get_variable("gamma"), tf.get_variable("beta")
            moving_avg, moving_var = tf.get_variable("moving_avg"), tf.get_variable("moving_var")
            shape = x.get_shape().as_list()
            control_inputs = []
            if train:
                avg, var = tf.nn.moments(x, range(len(shape)-1))
                update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
                update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                control_inputs = [update_moving_avg, update_moving_var]
            else:
                avg = moving_avg
                var = moving_var
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
    return output


def pool_layer(x, method='max', kernel=2, stride=2, padding='SAME'):
    # Perform a down sampling layer computation - "max" : max pooling, "avg" : avg pooling
    if method=="max":
        return tf.nn.max_pool(x, [1,kernel,kernel,1], [1,stride,stride,1], padding=padding)
    elif method=='avg':
        return tf.nn.avg_pool(x, [1,kernel,kernel,1], [1,stride,stride,1], padding=padding)
    else:
        raise ValueError

def full_layer(x, scope):
    # Perform a fully connected layer computation
    with tf.variable_scope(scope, reuse=True):
        w = tf.get_variable("wt")
        b = tf.get_variable("bi")
        fc = tf.matmul(x,w) + b
    return fc

def conv(x, scope, train, batch_norm):
    x = conv_layer(x, scope)
    if batch_norm:
        x = BatchNorm_layer(x, scope, train)
        #x = tf.contrib.layers.batch_norm(x, is_training=train, scope=scope, reuse=True)
    x = tf.nn.relu(x)
    x = pool_layer(x, "max")
    return x

def fc(x, scope, batch_norm, train, dropout, keep_prob=None):
    x = full_layer(x, scope)
    if batch_norm:
        #x = tf.contrib.layers.batch_norm(x, is_training=train, scope=scope, reuse=True)
        x = BatchNorm_layer(x, scope, train)
    x = tf.nn.relu(x)
    if dropout:
        x = tf.nn.dropout(x, keep_prob)
    return x

def convnet_stack(data, scopes, keep_prob, train=True, batch_norm=False):
    # Linearly Stacked CNN
    x = data
    # CONV Layers 1~3
    for scope in scopes[:3]:
        x = conv(x, scope, train, batch_norm)
    # FC Layer 4
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    x = fc(x, scopes[3], batch_norm, train, True, keep_prob)
    # FC Layer 5 - Output layer - no dropout
    x = fc(x, scopes[4], batch_norm, train, False)
    return x

def convnet_inception(data, scopes, keep_prob, train=True, batch_norm=False):
    x = data
    # Layer1 - shared conv layer
    x = conv(x, scope[0], train, batch_norm)
    # Layer2 - Top conv layer
    x_top = conv(x, scope[1], train, batch_norm)
    # Layer2 - Bottom conv layer
    x_bot = conv(x, scope[2], train, batch_norm)
    # Layer3 - Concatenate layers
    concat = tf.concat(3, [x_top, x_bot])
    x_pool = pool_layer(concat, "avg")
    # Layer4 - Fully connected layer1
    shape = x_pool.get_shape().as_list()
    x = tf.reshape(x_pool, [shape[0], -1])
    x = fc(x, scope[3], batch_norm, train, True, keep_prob)
    # Layer5 - Fully connected layer2 - output layer
    x = fc(x, scope[4], batch_norm, train, False)
    return x


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
        batch_norm = hyperparams['batch_norm']
        initialize_variables(convnet_shapes, initializer=hyperparams['initializer'], batch_norm=batch_norm)

        # Unwrap HyperParameters
        l2_reg = hyperparams['l2_reg'] # regularization penality factor
        tf_keep_prob, tfoptimizer = tf.placeholder(tf.float32), hyperparams['optimizer']
        init_lr,  global_step = hyperparams['init_lr'], tf.Variable(0)
        decay_steps, decay_rate = hyperparams['decay_steps'], hyperparams['decay_rate']
        learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate, staircase=True)
        
        # Compute Loss Function and Predictions
        train_logits = model(tf_train_dataset, scopes, tf_keep_prob, True, batch_norm)
        # Without regularization
        train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
        # With L2 regularization applied to fully connected layers
        l2_reg_loss = 0
        with tf.variable_scope('fc1', reuse=True):
            l2_reg_loss += tf.nn.l2_loss(tf.get_variable('wt')) + tf.nn.l2_loss(tf.get_variable('bi'))
        with tf.variable_scope('fc2', reuse=True):
            l2_reg_loss += tf.nn.l2_loss(tf.get_variable('wt')) + tf.nn.l2_loss(tf.get_variable('bi'))
        train_loss_l2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels) + l2_reg*l2_reg_loss)
        train_prediction = tf.nn.softmax(train_logits)
        # Optimizer
        optimizer = tfoptimizer(learning_rate).minimize(train_loss_l2, global_step=global_step)

        valid_logits = model(tf_valid_dataset, scopes, tf_keep_prob, False, batch_norm)
        valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits,tf_valid_labels))
        valid_prediction = tf.nn.softmax(valid_logits)
        if tf_test_dataset!=None:
            test_prediction = tf.nn.softmax(model(tf_test_dataset, scopes, tf_keep_prob, False, batch_norm))
        else:
            test_prediction = None
    
    # Train Convnet
    num_steps = epoches
    train_losses, valid_losses = np.zeros(num_steps), np.zeros(num_steps)
    train_acc, valid_acc = np.zeros(num_steps), np.zeros(num_steps)
    keep_prob = hyperparams['keep_prob']
    
    print "Start training", '.'*32
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            t = time.time()
            # Handle MiniBatch
            if minibatch:
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                #offset = np.random.randint(train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset+batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset+batch_size), :]
                feed_dict = {tf_train_dataset:batch_data, tf_train_labels:batch_labels, tf_keep_prob:keep_prob}
            else:
                feed_dict = {tf_keep_prob:keep_prob}
            # Run session...
            _, tl, predictions = session.run([optimizer, train_loss, train_prediction], feed_dict=feed_dict)
            train_losses[step] = tl
            if minibatch:
                train_acc[step] = accuracy(predictions, batch_labels)
            else:
                train_acc[step] = accuracy(predictions, tf_train_labels.eval(feed_dict={tf_keep_prob:1.0}))
            # Compute validation set accuracy
            valid_losses[step] = valid_loss.eval(feed_dict={tf_keep_prob:1.0})
            valid_acc[step] = accuracy(valid_prediction.eval(feed_dict={tf_keep_prob:1.0}), tf_valid_labels.eval())
            print('Epoch: %d:\tLoss: %f\t\tTime cost: %1.f\t\tTrain Acc: %.2f%%\tValid Acc: %.2f%%\tLearning rate: %.6f' \
                %(step, tl, (time.time()-t), (train_acc[step]*100), (valid_acc[step]*100),learning_rate.eval(),))
        print "Finished training", '.'*32
        # Compute test set accuracy
        if test_prediction!=None:
            test_acc = accuracy(test_prediction.eval(feed_dict={tf_keep_prob:1.0}), tf_test_labels.eval())
            print("Test accuracy: %.2f%%" %(test_acc*100))
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
    num_filter = 16
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
    hyperparams = {'keep_prob': 0.33, 'init_lr': 0.002, 'decay_rate': .9, 'decay_steps': 100,
                   'optimizer': tfoptimizer, 'l2_reg': 0.096, 'batch_norm': True,
                   'initializer': tf.truncated_normal_initializer(stddev=.015)} #tf.contrib.layers.variance_scaling_initializer()}

    # Setup computation graph and train convnet
    steps = 2501
    model, save_data_name = convnet_stack, 'training_data_stack3.1'
    #model, save_data_name = convnet_inception, 'training_data_inception'
    _, training_data = train_convnet(graph, model, tf_data, convnet_shapes, hyperparams, \
                                     steps, True, train_dataset, train_labels, batch_size)

    # Save data
    with open(save_data_name, 'w') as fh:
        pickle.dump(training_data, fh)



