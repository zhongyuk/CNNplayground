import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

# load data, prepare data
mnist = fetch_mldata('MNIST original', data_home='./')
print "MNIST data, X shape\t", mnist.data.shape
print "MNIST data, y shape\t", mnist.target.shape

print mnist.data.dtype
print mnist.target.dtype

mnist_X = mnist.data.astype(np.float32)
mnist_y = mnist.target.astype(np.float32)
print mnist_X.dtype
print mnist_y.dtype

# One-Hot-Encode y
num_classes = 10
mnist_y = np.arange(num_classes)==mnist_y[:, None]
mnist_y = mnist_y.astype(np.float32)
print mnist_y.shape

#Split training, validation, testing data
train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=10000,\
                                                      random_state=102, stratify=mnist.target)
train_X, test_X,  train_y, test_y  = train_test_split(train_X, train_y, test_size=10000,\
                                                     random_state=325, stratify=train_y)
print 'Dataset\t\tFeatureShape\tLabelShape'
print 'Training set:\t', train_X.shape,'\t', train_y.shape
print 'Validation set:\t', valid_X.shape,'\t', valid_y.shape
print 'Testing set:\t', test_X.shape, '\t', test_y.shape

#Build a simple 2 layer neural network graph
num_features = train_X.shape[1]
batch_size = 64
hidden_layer_size = 1024

# An initialization function
def initialize(scope, shape, wt_initializer, center=True, scale=True):
    with tf.variable_scope(scope, reuse=None) as sp:
        wt = tf.get_variable("weights", shape, initializer=wt_initializer)
        bi = tf.get_variable("biases", shape[-1], initializer=tf.constant_initializer(1.))
        if center:
            beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0))
        if scale:
            gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
        moving_avg = tf.get_variable("moving_mean", shape[-1], initializer=tf.constant_initializer(0.0), \
                                     trainable=False)
        moving_var = tf.get_variable("moving_variance", shape[-1], initializer=tf.constant_initializer(1.0), \
                                     trainable=False)
        sp.reuse_variables()

# Build Graph
init_lr = 0.001
graph = tf.Graph()
with graph.as_default():
    # prepare input tensor
    tf_train_X = tf.placeholder(tf.float32, shape=[batch_size, num_features])
    tf_train_y = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
    tf_valid_X, tf_valid_y = tf.constant(valid_X), tf.constant(valid_y)
    tf_test_X,  tf_test_y  = tf.constant(test_X),  tf.constant(test_y)
    
    # setup layers
    layers = [{'scope':'hidden_layer', 'shape':[num_features, hidden_layer_size], 
               'initializer':tf.truncated_normal_initializer(stddev=0.01)},
              {'scope':'output_layer', 'shape':[hidden_layer_size, num_classes],
               'initializer':tf.truncated_normal_initializer(stddev=0.01)}]
    # initialize layers
    for layer in layers:
        initialize(layer['scope'], layer['shape'], layer['initializer'])
    
    # build model - for each layer: -> X -> X*wt+bi -> batch_norm -> activation -> dropout (if not output layer) ->
    layer_scopes = [layer['scope'] for layer in layers]
    def model(X, layer_scopes, is_training, keep_prob, decay=0.9):
        output_X = X
        for scope in layer_scopes:
            # X*wt+bi
            with tf.variable_scope(scope, reuse=True):
                wt = tf.get_variable("weights")
                bi = tf.get_variable("biases")
            output_X = tf.matmul(output_X, wt) + bi
            # Insert Batch Normalization
            output_X = tf.contrib.layers.batch_norm(output_X, decay=decay, is_training=is_training,
            										center=True, scale=False,
                                                   updates_collections=None, scope=scope, reuse=True)
            # ReLu activation
            output_X = tf.nn.relu(output_X)
            # Dropout for all non-output layers
            if scope!=layer_scopes[-1]:
                output_X = tf.nn.dropout(output_X, keep_prob)
        return output_X
    
    # setup keep_prob
    keep_prob = tf.placeholder(tf.float32)
    
    # compute loss, make predictions
    train_logits = model(tf_train_X, layer_scopes, True, keep_prob)
    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_y))
    train_pred = tf.nn.softmax(train_logits)
    valid_logits = model(tf_valid_X, layer_scopes, False, keep_prob)
    valid_pred = tf.nn.softmax(valid_logits)
    test_logits = model(tf_test_X, layer_scopes, False, keep_prob)
    test_pred = tf.nn.softmax(test_logits)
    
    # compute accuracy
    def compute_accuracy(predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy
    
    train_accuracy = compute_accuracy(train_pred, tf_train_y)
    valid_accuracy = compute_accuracy(valid_pred, tf_valid_y)
    test_accuracy  = compute_accuracy(test_pred , tf_test_y)
    
    # setup learning rate, optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(init_lr,global_step, decay_steps=500, decay_rate=0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_loss, global_step=global_step)

# Start a session
num_steps = 1000
with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_y.shape[0] - batch_size)
        batch_X = train_X[offset:(offset+batch_size), :]
        batch_y = train_y[offset:(offset+batch_size), :]
        feed_dict = {tf_train_X : batch_X, tf_train_y : batch_y, keep_prob : 0.6}
        _, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy], feed_dict=feed_dict)
        vacc = sess.run(valid_accuracy, feed_dict={keep_prob : 1.0})
        if step%50==0:
            print('Epoch: %d:\tLoss: %f\t\tTrain Acc: %.2f%%\tValid Acc: %.2f%%\tLearning rate: %.6f' \
                %(step, tloss, (tacc*100), (vacc*100), learning_rate.eval()))
    print("Finished training")
    tacc = sess.run(test_accuracy, feed_dict={keep_prob : 1.0})
    print("Test accuracy: %.2f%%" %(tacc*100))