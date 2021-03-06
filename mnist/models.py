import tensorflow as tf
import numpy as np
import sys
from utils import *
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/")
from cnn import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def snn_f2(train_X, train_y, test_X, test_y):
	"""
	A Shallow Neuron Network: 2 fully connected layers
	Architecture: input -> fc1 -> BN -> ReLu -> dropout ->
						-> fc2 -> BN -> ReLu -> softmax -> output
	"""
	training_steps = 10001
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
		model.set_kp_value('fc1/dropout', 0.5)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(training_steps):
			offset = (step*batch_size)%(train_X.shape[0]-batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X,
									model.train_y : batch_y})
			_, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
							 feed_dict=train_feed_dict)
			if test_y is not None:
				if step%50==0:
					vacc = sess.run(valid_accuracy, feed_dict=model.kp_reference_feed_dict)
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
						%(step, tloss, (tacc*100), (vacc*100)))
			else:
				if step%50==0:
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
						%(step, tloss, (tacc*100)))
		print("Finished training")
		print("Making prediction.")
		prediction = sess.run(pred_y, feed_dict=model.kp_reference_feed_dict)
		print("Done making prediction.")
	return prediction

def cnn_c2f2(train_X, train_y, test_X, test_y):
	"""
	A Convolutional Neuron Network: 2 convolutional layers + 2 fully connected layers
	Architecture: input -> conv1 -> BN -> ReLu -> max pool ->
						-> conv2 -> BN -> ReLu -> max pool ->
						-> fc1   -> BN -> ReLu -> dropout  -> 
						-> fc2   -> BN -> ReLu -> softmax  -> output
	"""
	training_steps = 10001
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
		model.set_kp_value('fc1/dropout', 0.67)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(training_steps):
			offset = (step*batch_size)%(train_X.shape[0]-batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X,
									model.train_y : batch_y})
			_, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
							 feed_dict=train_feed_dict)
			if test_y is not None:
				if step%100==0:
					vacc = sess.run(valid_accuracy, feed_dict=model.kp_reference_feed_dict)
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
						%(step, tloss, (tacc*100), (vacc*100)))
			else:
				if step%100==0:
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
						%(step, tloss, (tacc*100)))
		print("Finished training")
		print("Making prediction.")
		prediction = sess.run(pred_y, feed_dict=model.kp_reference_feed_dict)
		print("Done making prediction.")
	return prediction


def cnn_c4f3(train_X, train_y, test_X, test_y):
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
	training_steps = 50001
	train_X = reshape_data(train_X)
	test_X = reshape_data(test_X)

	batch_size = 64
	input_shape = [batch_size, 28, 28, 1]
	conv_depth = 2
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
		conv_depth *= 2

	fc_wt_initializer = tf.contrib.layers.variance_scaling_initializer()
	fc_layers = [('fc1', 1024), ('fc2', 1024), ('fc3', num_class)]
	for fc_layer in fc_layers:
		layer_name, num_neuron = fc_layer[0], fc_layer[1]
		model.add_fc_layer(layer_name, num_neuron, 
						fc_wt_initializer, add_output_summary=False)
		model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=False)
		model.add_act_layer(layer_name+'/activation')
		if layer_name!='fc3':
			model.add_dropout_layer(layer_name+'/dropout')

	model.setup_learning_rate(0.01, exp_decay=True, decay_steps=5000, \
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
		model.set_kp_value('fc1/dropout', 0.4)
		model.set_kp_value('fc2/dropout', 0.7)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(training_steps):
			offset = (step*batch_size)%(train_X.shape[0]-batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X,
									model.train_y : batch_y})
			_, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
							 feed_dict=train_feed_dict)
			if test_y is not None:
				if step%100==0:
					vacc = sess.run(valid_accuracy, feed_dict=model.kp_reference_feed_dict)
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
						%(step, tloss, (tacc*100), (vacc*100)))
			else:
				if step%100==0:
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
						%(step, tloss, (tacc*100)))
		print("Finished training")
		print("Making prediction.")
		prediction = sess.run(pred_y, feed_dict=model.kp_reference_feed_dict)
		print("Done making prediction.")
	return prediction


def svm_model(train_X, train_y, test_X, test_y):
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
		print('Train Acc: %.2f%%\tValid Acc: %.2f%%' \
			%((train_accuracy*100), (valid_accuracy*100)))
	else:
		print('Train Acc: %.2f%%' %((train_accuracy*100)))
	print("Finished training")
	print("Making perdiction.")
	prediction = clf_obj.predict(test_X)
	return prediction 

def train_model(model, cnn_mode=True):
	"""
	A function for tuning any one of the models defined above in the models.py
	set cnn_mode to be False for snn_f2 model and svm_model
	"""
	train_filename = "/Users/Zhongyu/Documents/projects/kaggle/mnist/train.csv"
	X, y = load_data(train_filename)
	train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=2000,
										 random_state=263, stratify=y)
	if cnn_mode:
		train_X = reshape_data(train_X)
		valid_X = reshape_data(valid_X)
	pred = model(train_X, train_y, valid_X, valid_y)
	return pred

def cnn_c3f2(train_X, train_y, test_X, test_y):
	"""
	A Convolutional Neuron Network: 2 convolutional layers + 2 fully connected layers
	Architecture: input -> conv1 -> BN -> ReLu -> max pool ->
						-> conv2 -> BN -> ReLu -> max pool ->
						-> conv3 -> BN -> ReLu -> max pool ->
						-> fc1   -> BN -> ReLu -> dropout  ->
						-> fc2   -> BN -> ReLu -> dropout  -> output
	"""
	training_steps = 50001
	train_X = reshape_data(train_X)
	test_X = reshape_data(test_X)

	batch_size = 64
	input_shape = [batch_size, 28, 28, 1]
	conv_depth = 2
	num_class = 10

	# Build a ConvNet graph
	model = cnn_graph(input_shape, num_class)
	model.setup_data(batch_size, valid_X=test_X, valid_y=test_y)
	conv_wt_initializer = tf.truncated_normal_initializer(stddev=0.15)
	conv_layers = [('conv1', 5), ('conv2', 5), ('conv3', 5)]

	for conv_layer in conv_layers:
		layer_name, filter_size = conv_layer[0], conv_layer[1]
		model.add_conv2d_layer(layer_name, filter_size, conv_depth,
							conv_wt_initializer, add_output_summary=False)
		model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=False)
		model.add_act_layer(layer_name+'/activation')
		model.add_pool_layer(layer_name+'/pool')
		conv_depth *= 2

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

	model.setup_learning_rate(0.01, exp_decay=True, decay_steps=5000, \
							decay_rate=0.5, staircase=True, add_output_summary=False)

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
		model.set_kp_value('fc1/dropout', 0.6)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(training_steps):
			offset = (step*batch_size)%(train_X.shape[0]-batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X,
									model.train_y : batch_y})
			_, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
							 feed_dict=train_feed_dict)
			if test_y is not None:
				if step%100==0:
					vacc = sess.run(valid_accuracy, feed_dict=model.kp_reference_feed_dict)
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
						%(step, tloss, (tacc*100), (vacc*100)))
			else:
				if step%100==0:
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
						%(step, tloss, (tacc*100)))
		print("Finished training")
		print("Making prediction.")
		prediction = sess.run(pred_y, feed_dict=model.kp_reference_feed_dict)
		print("Done making prediction.")
	return prediction

def cnn_c2f2nin(train_X, train_y, test_X, test_y):
	"""
	A Convolutional Neuron Network: 2 convolutional layers + 2 fully connected layers
	Architecture: input -> conv1 -> BN -> ReLu -> max pool ->
						-> conv2 -> BN -> ReLu -> max pool -> (a conv2 performs a Network In Network 1x1 conv)
						-> fc1   -> BN -> ReLu -> dropout  -> 
						-> fc2   -> BN -> ReLu -> softmax  -> output
	"""
	training_steps = 10001
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
	conv_layers = [('conv1', 3), ('conv2', 1)]

	for conv_layer in conv_layers:
		layer_name, filter_size = conv_layer[0], conv_layer[1]
		model.add_conv2d_layer(layer_name, filter_size, conv_depth,
							conv_wt_initializer, add_output_summary=False)
		model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=False)
		model.add_act_layer(layer_name+'/activation')
		model.add_pool_layer(layer_name+'/pool')
		conv_depth = conv_depth/2

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
		model.set_kp_value('fc1/dropout', 0.67)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(training_steps):
			offset = (step*batch_size)%(train_X.shape[0]-batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X,
									model.train_y : batch_y})
			_, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
							 feed_dict=train_feed_dict)
			if test_y is not None:
				if step%100==0:
					vacc = sess.run(valid_accuracy, feed_dict=model.kp_reference_feed_dict)
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
						%(step, tloss, (tacc*100), (vacc*100)))
			else:
				if step%100==0:
					print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
						%(step, tloss, (tacc*100)))
		print("Finished training")
		print("Making prediction.")
		prediction = sess.run(pred_y, feed_dict=model.kp_reference_feed_dict)
		print("Done making prediction.")
	return prediction

def compare_models():
	"""
	compare cnn_c2f2 with cnn_c2f2nin, the latter used 1x1 convolution network in network structure
	with the exact same setting, cnn_c2f2 out performs cnn_c2f2nin by a small edge
	"""
	model1 = cnn_c2f2
	model2 = cnn_c2f2nin
	pred1 = train_model(model1, cnn_mode=True)
	pred2 = train_model(model2, cnn_mode=True)

def cnn_incept1f2(train_X, train_y, test_X, test_y):
    """
    A Convolutional Neuron Network: 1 inception convolutional layer + 2 fully connected layers
    Architecture: input -> convIncept(1x1 | 3x3 | 5x5) 	-> BN -> ReLu -> max pool ->
                        -> fc1   						-> BN -> ReLu -> dropout  -> 
                        -> fc2  						-> BN -> ReLu -> softmax  -> output
    """
    training_steps = 10001
    train_X = reshape_data(train_X)
    test_X = reshape_data(test_X)

    batch_size = 64
    input_shape = [batch_size, 28, 28, 1]
    conv_depth = 2
    num_class = 10

    # Build a ConvNet graph
    model = cnn_graph(input_shape, num_class)
    model.setup_data(batch_size, valid_X=test_X, valid_y=test_y)
    conv_wt_initializer = tf.truncated_normal_initializer(stddev=0.15)
    conv_layers = [('conv1', 5)]

    for conv_layer in conv_layers:
        layer_name, filter_size = conv_layer[0], conv_layer[1]
        model.add_convIncept_layer(layer_name, conv_depth, 
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
        model.set_kp_value('fc1/dropout', 0.67)
        train_feed_dict = model.get_kp_collection_dict()
        for step in range(training_steps):
            offset = (step*batch_size)%(train_X.shape[0]-batch_size)
            batch_X = train_X[offset:(offset+batch_size), :]
            batch_y = train_y[offset:(offset+batch_size), :]
            train_feed_dict.update({model.train_X : batch_X,
                                    model.train_y : batch_y})
            _, tloss, tacc = sess.run([optimizer, train_loss, train_accuracy],
                             feed_dict=train_feed_dict)
            if test_y is not None:
                if step%200==0:
                    vacc = sess.run(valid_accuracy, feed_dict=model.kp_reference_feed_dict)
                    print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
                        %(step, tloss, (tacc*100), (vacc*100)))
            else:
                if step%200==0:
                    print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\t' \
                        %(step, tloss, (tacc*100)))
        print("Finished training")
        print("Making prediction.")
        prediction = sess.run(pred_y, feed_dict=model.kp_reference_feed_dict)
        print("Done making prediction.")
    return prediction

if __name__=='__main__':
	# Run train_model func on cnn_c4f3 to tune the model
	import time
	t = time.time()
	model = cnn_c3f2
	pred = train_model(model, cnn_mode=True)
	print("time costs: %.2f" % (time.time()-t))
	#compare_models()

