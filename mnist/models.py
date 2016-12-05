import tensorflow as tf
import numpy as np
import sys
from utils import *
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/")
from cnn import *
from sklearn.svm import SVC

def snn_f2(train_X, train_y, test_X, test_y, training_steps):
    """
    A Shallow Neuron Network: 2 fully connected layers
    Architecture: input -> fc1 -> BN -> ReLu -> dropout ->
                        -> fc2 -> BN -> ReLu -> softmax -> output
    """
    batch_size = 64
    input_shape = [batch_size, train_X.shape[1]]
    num_class = 10
    model = cnn_graph(input_shape, num_class)
    model.setup_data(batch_size, valid_X=test_X, valid_y=test_y)
    wt_initializer = tf.contrib.layers.variance_scaling_initializer()
    layers = [('fc1', 1024), ('fc2', num_class)]
    for layer in layers:
        layer_name, num_neuron = layer[0], layer[1]
        model.add_fc_layer(layer_name, num_neuron,
                        wt_initializer, add_output_summary=False)
        model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=False)
        model.add_act_layer(layer_name+'/activation')
        if layer_name!='fc2':
            model.add_dropout_layer(layer_name+'/dropout')
    model.setup_learning_rate(0.01, exp_decay=True, decay_steps=200, \
                            decay_rate=0.8, staircase=False, add_output_summary=False)
    train_loss = model.compute_train_loss(add_output_summary=False)
    train_accuracy = model.evaluation("train", add_output_summary=False)
    if test_y is not None:
        valid_accuracy = model.evaluation("valid", add_output_summary=False)
    pred_y = model.make_prediction(test_X)

    optimizer = model.setup_optimizer(tf.train.AdamOptimizer, add_output_summary=False)
    graph = model.get_graph()

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(training_steps):
            offset = (step*batch_size)%(train_X.shape[0]-batch_size)
            batch_X = train_X[offset:(offset+batch_size), :]
            batch_y = train_y[offset:(offset+batch_size), :]
            feed_dict = {model.train_X : batch_X,
                         model.train_y : batch_y,
                         model.keep_probs[0] : 0.6}
            _, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
                             feed_dict=feed_dict)
            if test_y is not None:
                feed_dict[model.keep_probs[0]] = 1.0
                vacc = sess.run(valid_accuracy, feed_dict=feed_dict)
                if step%50==0:
                    print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
                        %(step, tloss, (tacc*100), (vacc*100)))
            else:
                if step%50==0:
                    print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
                        %(step, tloss, (tacc*100)))
        print("Finished training")
        print("Making prediction.")
        prediction = sess.run(pred_y, feed_dict={model.keep_probs[0]:1.0})
        print("Done making prediction.")
    return prediction

def cnn_c2f2(train_X, train_y, test_X, test_y, training_steps):
    """
    A Convolutional Neuron Network: 2 convolutional layers + 2 fully connected layers
    Architecture: input -> conv1 -> BN -> ReLu -> max pool ->
                        -> conv2 -> BN -> ReLu -> max pool ->
                        -> fc1   -> BN -> ReLu -> dropout  -> 
                        -> fc2   -> BN -> ReLu -> softmax  -> output
    """
    train_X = reshape_data(train_X)
    test_X = reshape_data(test_X)

    batch_size = 64
    input_shape = [batch_size, 28, 28, 1]
    conv_depth = 4
    num_class = 10

    # Build a ConvNet graph
    model = cnn_graph(input_shape, num_class)
    model.setup_data(batch_size, valid_X=test_X, valid_y=test_y)
    conv_wt_initializer = tf.truncated_normal_initializer(stddev=0.15)
    conv_layers = [('conv1', 3), ('conv2', 3)]

    for conv_layer in conv_layers:
        layer_name, filter_size = conv_layer[0], conv_layer[1]
        model.add_conv2d_layer(layer_name, filter_size, conv_depth,
                            conv_wt_initializer, add_output_summary=False)
        model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=False)
        model.add_act_layer(layer_name+'/activation')
        model.add_pool_layer(layer_name+'/pool')

    fc_wt_initializer = tf.contrib.layers.variance_scaling_initializer()
    fc_layers = [('fc1', 1024), ('fc2', num_class)]
    for fc_layer in fc_layers:
        layer_name, num_neuron = fc_layer[0], fc_layer[1]
        model.add_fc_layer(layer_name, num_neuron, 
                        fc_wt_initializer, add_output_summary=False)
        model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=False)
        model.add_act_layer(layer_name+'/activation')
        if layer_name!='fc2':
            model.add_dropout_layer(layer_name+'/dropout')

    model.setup_learning_rate(0.01, exp_decay=True, decay_steps=1000, \
                            decay_rate=0.5, staircase=False, add_output_summary=False)

    train_loss = model.compute_train_loss(add_output_summary=False)
    train_accuracy = model.evaluation("train", add_output_summary=False)
    if test_y is not None:
        valid_accuracy = model.evaluation("valid", add_output_summary=False)
    pred_y = model.make_prediction(test_X)

    optimizer = model.setup_optimizer(tf.train.AdamOptimizer, add_output_summary=False)
    graph = model.get_graph()

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(training_steps):
            offset = (step*batch_size)%(train_X.shape[0]-batch_size)
            batch_X = train_X[offset:(offset+batch_size), :]
            batch_y = train_y[offset:(offset+batch_size), :]
            feed_dict = {model.train_X : batch_X,
                         model.train_y : batch_y,
                         model.keep_probs[0] : 0.67}
            _, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
                             feed_dict=feed_dict)
            if test_y is not None:
                feed_dict[model.keep_probs[0]] = 1.0
                vacc = sess.run(valid_accuracy, feed_dict=feed_dict)
                if step%50==0:
                    print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
                        %(step, tloss, (tacc*100), (vacc*100)))
            else:
                if step%50==0:
                    print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
                        %(step, tloss, (tacc*100)))
        print("Finished training")
        print("Making prediction.")
        prediction = sess.run(pred_y, feed_dict={model.keep_probs[0]:1.0})
        print("Done making prediction.")
    return prediction


def cnn_c4f3(train_X, train_y, test_X, test_y, training_steps):
    """
    A Convolutional Neuron Network: 2 convolutional layers + 2 fully connected layers
    Architecture: input -> conv1 -> BN -> ReLu -> max pool ->
                        -> conv2 -> BN -> ReLu -> max pool ->
                        -> conv3 -> BN -> ReLu -> max pool ->
                        -> conv4 -> BN -> ReLu -> max pool ->
                        -> fc1   -> BN -> ReLu -> dropout  ->
                        -> fc2   -> BN -> ReLu -> dropout  -> 
                        -> fc3   -> BN -> ReLu -> softmax  -> output
    """
    train_X = reshape_data(train_X)
    test_X = reshape_data(test_X)

    batch_size = 64
    input_shape = [batch_size, 28, 28, 1]
    conv_depth = 4
    num_class = 10

    # Build a ConvNet graph
    model = cnn_graph(input_shape, num_class)
    model.setup_data(batch_size, valid_X=test_X, valid_y=test_y)
    conv_wt_initializer = tf.truncated_normal_initializer(stddev=0.15)
    conv_layers = [('conv1', 3), ('conv2', 3), ('conv3', 3), ('conv4', 3)]

    for conv_layer in conv_layers:
        layer_name, filter_size = conv_layer[0], conv_layer[1]
        model.add_conv2d_layer(layer_name, filter_size, conv_depth,
                            conv_wt_initializer, add_output_summary=False)
        model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=False)
        model.add_act_layer(layer_name+'/activation')
        model.add_pool_layer(layer_name+'/pool')

    fc_wt_initializer = tf.contrib.layers.variance_scaling_initializer()
    fc_layers = [('fc1', 1024), ('fc2', 512), ('fc3', num_class)]
    for fc_layer in fc_layers:
        layer_name, num_neuron = fc_layer[0], fc_layer[1]
        model.add_fc_layer(layer_name, num_neuron, 
                        fc_wt_initializer, add_output_summary=False)
        model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=False)
        model.add_act_layer(layer_name+'/activation')
        if layer_name!='fc3':
            model.add_dropout_layer(layer_name+'/dropout')

    model.setup_learning_rate(0.1, exp_decay=True, decay_steps=250, \
                            decay_rate=0.8, staircase=True, add_output_summary=False)

    train_loss = model.compute_train_loss(add_output_summary=False)
    train_accuracy = model.evaluation("train", add_output_summary=False)
    if test_y is not None:
        valid_accuracy = model.evaluation("valid", add_output_summary=False)
    pred_y = model.make_prediction(test_X)

    optimizer = model.setup_optimizer(tf.train.AdamOptimizer, add_output_summary=False)
    graph = model.get_graph()

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(training_steps):
            offset = (step*batch_size)%(train_X.shape[0]-batch_size)
            batch_X = train_X[offset:(offset+batch_size), :]
            batch_y = train_y[offset:(offset+batch_size), :]
            feed_dict = {model.train_X : batch_X,
                         model.train_y : batch_y,
                         model.keep_probs[0] : 0.4,
                         model.keep_probs[1] : 0.6}
            _, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
                             feed_dict=feed_dict)
            if test_y is not None:
                feed_dict[model.keep_probs[0]] = 1.0
                feed_dict[model.keep_probs[1]] = 1.0
                vacc = sess.run(valid_accuracy, feed_dict=feed_dict)
                if step%10==0:
                    print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
                        %(step, tloss, (tacc*100), (vacc*100)))
            else:
                if step%10==0:
                    print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
                        %(step, tloss, (tacc*100)))
        print("Finished training")
        print("Making prediction.")
        prediction = sess.run(pred_y, feed_dict={model.keep_probs[0]:1.0, model.keep_probs[1]:1.0})
        print("Done making prediction.")
    return prediction


def svm_model(train_X, train_y, test_X, test_y, training_steps):
    """
    A Support Vector Machine Model
    """
    train_label = np.argmax(train_y, 1)
    if test_y is not None:
        test_label  = np.argmax(test_y, 1)
    clf_obj = SVC(C=7, kernel='rbf', gamma='auto')
    clf_obj.fit(train_X, train_label)
    train_accuracy = clf_obj.score(train_X, train_label)
    if test_y is not None:
        valid_accuracy = clf_obj.score(test_X, test_label)
        print('Train Acc: %.2f%%\tValid Acc: %2.f%%') \
            %((train_accuracy*100), (valid_accuracy*100))
    else:
        print('Train Acc: %.2f%%' %((train_accuracy*100)))
    print("Finished training")
    print("Making perdiction.")
    prediction = clf_obj.predict(test_X)
    return prediction 

