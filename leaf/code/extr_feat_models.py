import tensorflow as tf
from preparation_pipes import *
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground")
from cnn import *

def MLP_model(train_X, train_y, test_X, test_y, epoches, 
			num_neuron, learning_rate, dropout, logdir):
	"""
	A simple Multi-Logistic-Preceptron model:
	Architecture: input -> fc1 (BN->ReLu->Dropout) ->
						-> fc2 (BN->ReLu) -> Softmax -> output
	"""
	batch_size = 64
	input_shape = [batch_size, train_X.shape[1]]
	num_class = train_y.shape[1]
	model = cnn_graph(input_shape, num_class)
	model.setup_data(batch_size, valid_X=test_X, valid_y=test_y)
	wt_initializer = tf.contrib.layers.xavier_initializer()
	layers = [('fc1', num_neuron), ('fc2', num_class)]
	for layer in layers:
		layer_name, num_neuron = layer[0], layer[1]
		model.add_fc_layer(layer_name, num_neuron, wt_initializer, add_output_summary=True)
		model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=True)
		model.add_act_layer(layer_name+'/activation')
		if layer_name!=layers[-1][0]:
			model.add_dropout_layer(layer_name+'/dropout')
	model.setup_learning_rate(learning_rate, exp_decay=False, add_output_summary=False)
	train_loss = model.compute_train_loss(add_output_summary=True)
	valid_loss = model.compute_valid_loss(add_output_summary=True)
	optimizer = model.setup_optimizer(tf.train.AdamOptimizer, add_output_summary=False)
	merged_summary = model.merge_summaries()
	with tf.Session(graph=model.get_graph()) as sess:
		print("Create summary writers")
		train_writer = tf.train.SummaryWriter('tmp/'+logdir+'/train', graph=sess.graph)
		valid_writer = tf.train.SummaryWriter('tmp/'+logdir+'/valid')
		tf.initialize_all_variables().run()
		print("Initialized")
		model.set_kp_value('fc1/dropout', dropout)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(epoches):
			offset = (step*batch_size)%(train_X.shape[0]-batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X, model.train_y : batch_y})
			_, tloss, tmrg_summ = sess.run([optimizer, train_loss, merged_summary], feed_dict=train_feed_dict)
			train_writer.add_summary(tmrg_summ, step)
			if step%500 == 0:
				valid_feed_dict = dict(model.kp_reference_feed_dict)
				valid_feed_dict.update({model.train_X : batch_X, model.train_y : batch_y})
				vloss, vmrg_summ = sess.run([valid_loss, merged_summary], feed_dict=valid_feed_dict)
				valid_writer.add_summary(vmrg_summ, step)
				print('Epoch: %d:\tTrain Loss: %.6f\tValid Loss: %.6f' %(step, tloss, vloss))
		print("Finished training")
		vloss = sess.run(valid_loss, feed_dict=valid_feed_dict)
		print("Final valid loss: %.6f" %(vloss))


def CNN_model(train_X, train_y, test_X, test_y, epoches, 
			num_neuron, learning_rate, dropout, logdir):
	"""
	A simple Convolutional-Neural-Network model:
	Architecture: input -> InceptConv1 (BN->ReLu->) ->
						-> fc1 (BN->ReLu->Dropout)  ->
						-> fc2 (BN->ReLu) -> Softmax -> output
	"""
	batch_size = 64
	input_shape = [batch_size, train_X.shape[1], train_X.shape[2], train_X.shape[3]]
	num_class = train_y.shape[1]
	model = cnn_graph(input_shape, num_class)
	model.setup_data(batch_size, valid_X=test_X, valid_y=test_y)
	conv_wt_initializer = tf.contrib.layers.xavier_initializer_conv2d()
	#conv_layer = ('IncepConv1', 5)
	# Convolutional layer
	#layer_name, filter_size = conv_layer[0], conv_layer[1]
	layer_name, conv_depth = 'IncepConv1', 3
	model.add_convIncept_layer(layer_name, conv_depth, conv_wt_initializer, add_output_summary=True)
	model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=True)
	model.add_act_layer(layer_name+'/activation')
	#model.add_pool_layer(layer_name+'/pool')

	# Fully Connected layer
	fc_layers = [('fc1', 3096), ('fc2', num_class)]
	fc_wt_initializer = tf.contrib.layers.variance_scaling_initializer()
	for fc_layer in fc_layers:
		layer_name, num_neuron = fc_layer[0], fc_layer[1]
		model.add_fc_layer(layer_name, num_neuron, fc_wt_initializer, add_output_summary=True)
		model.add_batchnorm_layer(layer_name+'/batchnorm', add_output_summary=True)
		model.add_act_layer(layer_name+'/activation')
		if layer_name!=fc_layers[-1][0]:
			model.add_dropout_layer(layer_name+'/dropout')

	model.setup_learning_rate(learning_rate, exp_decay=False, add_output_summary=False)
	train_loss = model.compute_train_loss(add_output_summary=True)
	valid_loss = model.compute_valid_loss(add_output_summary=True)
	optimizer = model.setup_optimizer(tf.train.AdamOptimizer, add_output_summary=False)
	merged_summary = model.merge_summaries()
	with tf.Session(graph=model.get_graph()) as sess:
		print("Create summary writers")
		train_writer = tf.train.SummaryWriter('tmp/'+logdir+'/train', graph=sess.graph)
		valid_writer = tf.train.SummaryWriter('tmp/'+logdir+'/valid')
		tf.initialize_all_variables().run()
		print("Initialized")
		model.set_kp_value('fc1/dropout', dropout)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(epoches):
			offset = (step*batch_size)%(train_X.shape[0]-batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X, model.train_y : batch_y})
			_, tloss, tmrg_summ = sess.run([optimizer, train_loss, merged_summary], feed_dict=train_feed_dict)
			train_writer.add_summary(tmrg_summ, step)
			if step%500 == 0:
				valid_feed_dict = dict(model.kp_reference_feed_dict)
				valid_feed_dict.update({model.train_X : batch_X, model.train_y : batch_y})
				vloss, vmrg_summ = sess.run([valid_loss, merged_summary], feed_dict=valid_feed_dict)
				valid_writer.add_summary(vmrg_summ, step)
				print('Epoch: %d:\tTrain Loss: %.6f\tValid Loss: %.6f' %(step, tloss, vloss))
		print("Finished training")
		vloss = sess.run(valid_loss, feed_dict=valid_feed_dict)
		print("Final valid loss: %.6f" %(vloss))

if __name__=='__main__':
	train_filename = '../train.csv'
	# Simple MLP model
	#train_X, train_y, test_X, test_y = simple_MLP_pipe(train_filename, 330)
	#MLP_model(train_X, train_y, test_X, test_y, 20000, 1024, 0.001, 0.5, 'simple')

	# PCA MLP model
	#train_X, train_y, test_X, test_y = pca_MLP_pipe(train_filename, 330)
	#MLP_model(train_X, train_y, test_X, test_y, 20000, 512, 0.001, 0.5, 'pca')
	
	# CNN model
	train_X, train_y, test_X, test_y = CNN_pipe(train_filename, 330)
	CNN_model(train_X, train_y, test_X, test_y, 20000, 3096, 0.0005, 0.3, 'cnn')