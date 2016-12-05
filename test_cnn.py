import tensorflow as tf
import numpy as np
from cnn import *
from tensorflow.python.framework import ops
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/cifar10/")
from prepare_input import *

def test_conv2d(steps):
    try:
        convx = conv2d('convx', [1,2])
    except ValueError:
        print("Successfully catch shape error")

    sess = tf.InteractiveSession()
    shape = [1, 1, 3, 1]
    conv1 = conv2d('conv1', shape) 
    assert(conv1.get_layer_name()=='conv1')
    assert(conv1.get_layer_type()=='2D Convolution Layer')
    assert(conv1.is_trainable()==True)
    assert(conv1.get_shape()==shape)
    conv1.initialize()
    conv1.add_variable_summaries()
    wt, bi = conv1.get_variables()
    sess.run([tf.initialize_all_variables()])
    wt_val, bi_val = sess.run([wt, bi])
    print("initialized weights:", wt_val)
    print("initialized biases:", bi_val)

    input = tf.placeholder(tf.float32, [4, 2, 2, 3])
    for i in range(steps):
        X_np = np.random.randn(4, 2, 2, 3)
        print('*'*16, i, '*'*16)
        y = conv1.train(input)
        print(y.eval(feed_dict={input:X_np}))
    sess.close()

def test_fc(steps):
    try:
        fcx = fc('fcx', [1,2,3,4])
    except ValueError:
        print("Successfully catch shape error")

    sess = tf.InteractiveSession()
    shape = [2, 1]
    fc1 = fc('fc1', shape)
    assert(fc1.get_layer_name()=='fc1')
    assert(fc1.get_layer_type()=='Fully Connected Layer')
    assert(fc1.is_trainable()==True)
    assert(fc1.get_shape()==shape)
    fc1.initialize()
    fc1.add_variable_summaries()
    wt, bi = fc1.get_variables()
    sess.run(tf.initialize_all_variables())
    wt_val, bi_val = sess.run([wt, bi])
    print("initialized weights: ", wt_val)
    print("initialized biases: ", bi_val)

    input = tf.placeholder(tf.float32, [1, 2])
    for i in range(steps):
        X_np = np.random.randn(1, 2)
        print('*'*16, i, '*'*16)
        y = fc1.train(input)
        print(y.eval(feed_dict={input:X_np}))
    sess.close()

def test_act(steps):
    sess = tf.InteractiveSession()
    act1 = activation('act1')
    assert(act1.get_layer_name()=='act1')
    assert(act1.get_layer_type()=='Activation Layer')
    assert(act1.is_trainable()==False)
    act1.get_act_func()
    act1.set_act_func(tf.nn.relu)
    assert(act1.get_act_func()=='relu')

    input = tf.placeholder(tf.float32, [2, 2])
    for i in range(steps):
        X_np = np.random.randn(2,2)
        print('*'*16, i, '*'*16)
        print(X_np)
        print("-"*32)
        y = act1.train(input)
        print(y.eval(feed_dict={input:X_np}))
    sess.close()

def test_pool2d(steps):
    sess = tf.InteractiveSession()
    pool1 = pool2d('pool1')
    assert(pool1.get_layer_name()=='pool1')
    assert(pool1.get_layer_type()=='2D Pooling Layer')
    assert(pool1.is_trainable()==False)
    pool1.get_pool_func()
    pool1.set_pool_func(tf.nn.max_pool, ksize=[1,2,2,1], \
                        strides=[1,2,2,1], padding='SAME')
    assert(pool1.get_pool_func()=='max_pool')

    input = tf.placeholder(tf.float32, [1,2,2,1])
    for i in range(steps):
        X_np = np.random.randn(1,2,2,1)
        print('*'*16, i, '*'*16)
        print(X_np)
        print('-'*32)
        y = pool1.train(input)
        print(y.eval(feed_dict={input : X_np}))

def test_dropout(steps):
    sess = tf.InteractiveSession()
    dropout1 = dropout('dropout1')
    assert(dropout1.get_layer_name()=='dropout1')
    assert(dropout1.get_layer_type()=='Dropout Layer')
    assert(dropout1.is_trainable()==False)

    input = tf.placeholder(tf.float32, [5, 2])
    for i in range(steps):
        X_np = np.random.randn(5,2)
        print("*"*16, i, "*"*16)
        print(X_np)
        print('-'*32)
        y = dropout1.train(input)
        print(y.eval(feed_dict={input : X_np, dropout1.keep_prob : .5}))


def test_batchnorm(steps):
    sess = tf.InteractiveSession()
    batchnorm1 = batchnorm('batchnorm1', 3)
    assert(batchnorm1.get_layer_name()=='batchnorm1')
    assert(batchnorm1.get_layer_type()=='Batch Normalization Layer')
    assert(batchnorm1.is_trainable()==True)
    batchnorm1.initialize()
    batchnorm1.add_variable_summaries()
    sess.run([tf.initialize_all_variables()])
    beta, gamma, moving_avg, moving_var = batchnorm1.get_variables()
    beta_val, gamma_val, mv_avg, mv_var = sess.run([beta, gamma, moving_avg, moving_var])
    print("initialized variables values", '='*8)
    print("offset_factor", '-'*16); print(beta_val)
    print("scale_factor", '-'*16); print(gamma_val)
    print("moving mean", "-"*16); print(mv_avg)
    print("moving variance", "-"*16); print(mv_var)
    print("="*32)

    input = tf.placeholder(tf.float32, [2,3])
    for i in range(steps):
        X_np = (i+1)*np.ones([2,3])
        print("*"*16, i, "*"*16)
        print(X_np)
        print('+'*16, 'is_training==True','+'*16)
        y = batchnorm1.train(input, True)
        beta, gamma, moving_avg, moving_var = batchnorm1.get_variables()
        loc_avg, loc_var = tf.nn.moments(input, [0])
        with ops.control_dependencies([loc_avg, loc_var, beta, gamma]):
            y_bn = tf.nn.batch_normalization(input, loc_avg, loc_var, beta,
                                             gamma, variance_epsilon=0.001)
        mv_avg, mv_var, y_val1, y_val2 = sess.run([moving_avg, moving_var, y, y_bn], 
                                                  feed_dict={input : X_np})
        print("moving mean", "-"*16); print(mv_avg)
        print("moving variance", "-"*16); print(mv_var)
        print("output",'-'*16); print(y_val1)
        assert(np.array_equal(y_val1, y_val2))

        print('+'*16, 'is_training==False','+'*16)
        y = batchnorm1.train(input, False)
        beta, gamma, moving_avg, moving_var = batchnorm1.get_variables()
        with ops.control_dependencies([beta, gamma, moving_avg, moving_var]):
            y_bn = tf.nn.batch_normalization(input, moving_avg, moving_var, beta,
                                             gamma, variance_epsilon=.001)
        mv_avg, mv_var, y_val1, y_val2 = sess.run([moving_avg, moving_var, y, y_bn],
                                                  feed_dict={input : X_np})
        print("moving mean", "-"*16); print(mv_avg)
        print("moving variance", "-"*16); print(mv_var)
        print("output", "-"*16); print(y_val1)
        assert(np.array_equal(y_val1, y_val2))


def test_cnn_graph(steps):
    data_dir = "./cifar10/data/"
    dataset_list = prepare_cifar10_input(data_dir)
    train_dataset, train_labels = dataset_list[0], dataset_list[1]
    valid_dataset, valid_labels = dataset_list[2], dataset_list[3]
    test_dataset , test_labels  = dataset_list[4], dataset_list[5]
    input_shape = [-1, 32, 32, 3]
    batch_size = 32
    conv_depth = 4
    cnn_model = cnn_graph(input_shape, 10)
    cnn_model.setup_data(batch_size, test_dataset, test_labels, valid_dataset, valid_labels)
    # Build CNN layers
    wt_initializer = tf.truncated_normal_initializer(stddev=.015)
    cnn_model.add_conv2d_layer("conv1", 3, conv_depth, wt_initializer)
    conv1_layer_shape = cnn_model.get_layers()[-1]['layer_shape']
    assert(conv1_layer_shape == [-1, 32, 32, 4])
    cnn_model.add_batchnorm_layer("conv1/batchnorm")
    conv1_bn_layer = cnn_model.get_layers()[-1]
    assert(conv1_bn_layer['layer_name']=='conv1/batchnorm')
    assert(conv1_bn_layer['layer_shape']==[-1, 32, 32, 4])
    cnn_model.add_act_layer("conv1/activation")
    conv1_act_layer = cnn_model.get_layers()[-1]
    assert(conv1_act_layer['layer_name']=='conv1/activation')
    assert(conv1_act_layer['layer_shape']==[-1, 32, 32, 4])
    cnn_model.add_pool_layer("conv1/pool")
    conv1_pool_layer = cnn_model.get_layers()[-1]
    assert(conv1_pool_layer['layer_name']=='conv1/pool')
    assert(conv1_pool_layer['layer_shape']==[-1, 16, 16, 4])
    cnn_model.add_conv2d_layer("conv2", 3, conv_depth, wt_initializer)
    conv2_layer_shape = cnn_model.get_layers()[-1]['layer_shape']
    assert(conv2_layer_shape==[-1, 16, 16, 4])
    cnn_model.add_batchnorm_layer("conv2/batchnorm")
    conv2_bn_layer = cnn_model.get_layers()[-1]
    assert(conv2_bn_layer['layer_name']=='conv2/batchnorm')
    assert(conv2_bn_layer['layer_shape']==[-1, 16, 16, 4])
    cnn_model.add_act_layer("conv2/activation")
    conv2_act_layer = cnn_model.get_layers()[-1]
    assert(conv2_act_layer['layer_name']=="conv2/activation")
    assert(conv2_act_layer['layer_shape']==[-1, 16, 16, 4])
    cnn_model.add_pool_layer("conv2/pool")
    conv2_pool_layer = cnn_model.get_layers()[-1]
    assert(conv2_pool_layer['layer_name']=='conv2/pool')
    assert(conv2_pool_layer['layer_shape']==[-1, 8, 8, 4])
    cnn_model.add_fc_layer("fc1", 64, wt_initializer)
    fc1_layer = cnn_model.get_layers()[-1]
    assert(fc1_layer['layer_shape']==[-1, 64])
    cnn_model.add_batchnorm_layer("fc1/batchnorm")
    fc1_bn_layer = cnn_model.get_layers()[-1]
    assert(fc1_bn_layer["layer_name"]=='fc1/batchnorm')
    assert(fc1_bn_layer['layer_shape']==[-1, 64])
    cnn_model.add_act_layer("fc1/activation")
    fc1_act_layer = cnn_model.get_layers()[-1]
    assert(fc1_act_layer['layer_name']=='fc1/activation')
    assert(fc1_act_layer["layer_shape"]==[-1, 64])
    cnn_model.add_dropout_layer("fc1/dropout")
    fc1_dropout_layer = cnn_model.get_layers()[-1]
    assert(fc1_dropout_layer['layer_name']=='fc1/dropout')
    assert(fc1_dropout_layer['layer_shape']==[-1, 64])
    cnn_model.add_fc_layer("fc2", 10, wt_initializer)
    fc2_layer = cnn_model.get_layers()[-1]
    assert(fc2_layer['layer_shape']==[-1, 10])
    cnn_model.add_act_layer("fc2/activation")
    fc2_act_layer = cnn_model.get_layers()[-1]
    assert(fc2_act_layer['layer_name']=="fc2/activation")
    assert(fc2_act_layer['layer_shape']==[-1, 10])
    
    print("done building up network...")
    cnn_model.setup_learning_rate(0.0003)
    train_loss = cnn_model.compute_train_loss(add_output_summary=False)
    valid_loss = cnn_model.compute_valid_loss()
    train_accuracy = cnn_model.evaluation("train")
    valid_accuracy = cnn_model.evaluation("valid")
    test_accuracy  = cnn_model.evaluation("test")
    optimizer = cnn_model.setup_optimizer(tf.train.AdamOptimizer)
    merged_summary = cnn_model.merge_summaries()
    graph = cnn_model.get_graph()
    with tf.Session(graph=graph) as sess:
        print("Create summary writers")
        train_writer = tf.train.SummaryWriter('tmp/' + '/train', graph=sess.graph)
        valid_writer = tf.train.SummaryWriter('tmp/' + '/valid')
        
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(steps):
            offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
            batch_X = train_dataset[offset:(offset+batch_size), :]
            batch_y = train_labels[offset:(offset+batch_size), :]
            feed_dict = {cnn_model.train_X : batch_X, 
                        cnn_model.train_y : batch_y,
                        cnn_model.keep_prob : 0.5}
            _, tloss, tacc, tmrg_summ = sess.run([optimizer, train_loss, train_accuracy,\
                                        merged_summary], feed_dict=feed_dict)
            train_writer.add_summary(tmrg_summ, step)
            vacc, vmrg_summ = sess.run([valid_accuracy, merged_summary], \
                              feed_dict={cnn_model.train_X : batch_X,
                                         cnn_model.train_y : batch_y,
                                         cnn_model.keep_prob : 1.0})
            valid_writer.add_summary(vmrg_summ, step)
            print('Epoch: %d:\tLoss: %f\t\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
                %(step, tloss, (tacc*100), (vacc*100)))
        tacc = sess.run(test_accuracy, feed_dict={cnn_model.keep_prob : 1.0})
        print("Finished training")
        print("Test accuracy: %.2f%%" %(tacc*100))


if __name__=='__main__':
    test_conv_bool = input("Test conv2d layer? [y] or [n]")
    if test_conv_bool=='y':
        test_conv2d(1)

    test_fc_bool = input("Test fc layer? [y] or [n]")
    if test_fc_bool=='y':
        test_fc(2)

    test_act_bool = input("Test activation layer? [y] or [n]")
    if test_act_bool=='y':
        test_act(2)

    test_pool_bool = input("Test pool2d layer? [y] or [n]")
    if test_pool_bool=='y':
        test_pool2d(1)

    test_dropout_bool = input("Test dropout layer? [y] or [n]")
    if test_dropout_bool=='y':
        test_dropout(2)

    test_batchnorm_bool =input("Test batchnorm layer? [y] or [n]")
    if test_batchnorm_bool=='y':
        test_batchnorm(2)

    test_cnn_graph_bool = input("Test cnn_graph? [y] or [n]")
    if test_cnn_graph_bool=='y':
        steps = int(input("How many training steps do you want to test run?"))
        test_cnn_graph(steps)
