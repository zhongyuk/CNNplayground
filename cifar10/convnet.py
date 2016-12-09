import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
from prepare_input import *
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/")
from cnn import *

def convnet_model(training_steps):
	# Prepare CIFAR10 data input
	data_dir = "./data/"
	dataset_list = prepare_cifar10_input(data_dir, True)
	train_dataset, train_labels = dataset_list[0], dataset_list[1]
	valid_dataset, valid_labels = dataset_list[2], dataset_list[3]
	test_dataset , test_labels  = dataset_list[4], dataset_list[5]

	batch_size = 32
	input_shape = [batch_size, 32, 32, 3]
	conv_depth = 4
	num_class = 10

	# Build a convnet graph
	model = cnn_graph(input_shape, num_class)
	model.setup_data(batch_size, test_dataset, test_labels, valid_dataset, valid_labels)
	conv_wt_initializer = tf.truncated_normal_initializer(stddev=.15)
	conv_layers = [('conv1', 3), ('conv2', 5), ('conv3', 3) ]

	for conv_layer in conv_layers:
		layer_name, filter_size = conv_layer[0], conv_layer[1]
		model.add_conv2d_layer(layer_name, filter_size, conv_depth,  conv_wt_initializer)
		model.add_batchnorm_layer(layer_name+"/batchnorm")
		model.add_act_layer(layer_name+"/activation")
		model.add_pool_layer(layer_name+"/pool")

	fc_wt_initializer = tf.contrib.layers.variance_scaling_initializer()
	fc_layers = [("fc1", 512), ('fc2', 256), ('fc3', num_class)]
	for fc_layer in fc_layers:
		layer_name, num_neuron = fc_layer[0], fc_layer[1]
		model.add_fc_layer(layer_name, num_neuron, fc_wt_initializer)
		model.add_batchnorm_layer(layer_name+"/batchnorm")
		model.add_act_layer(layer_name+"/activation")
		if layer_name!='fc3':
			model.add_dropout_layer(layer_name+"/dropout")

	model.setup_learning_rate(0.01, exp_decay=True, decay_steps=200, \
							 decay_rate=0.95, staircase=False,)

	train_loss = model.compute_train_loss(add_output_summary=False)
	valid_loss = model.compute_valid_loss()

	train_accuracy = model.evaluation("train")
	valid_accuracy = model.evaluation("valid")
	test_accuracy = model.evaluation("test")

	# with L2 regularization
	#l2_reg_factor = 0.1
	#optimizer = model.setup_optimizer(tf.train.AdamOptimizer, l2_reg=True, l2_reg_factor=l2_reg_factor)
	# without L2 regularization
	optimizer = model.setup_optimizer(tf.train.AdamOptimizer)
	learning_rate = model.get_learning_rate()
	merged_summary = model.merge_summaries()
	graph = model.get_graph()

	# variables need to be saved
	train_losses, valid_losses = np.zeros(training_steps), []
	train_acc, valid_acc = np.zeros(training_steps), []

	with tf.Session(graph=graph) as sess:
		print("Create summary writers")
		train_writer = tf.train.SummaryWriter('tmp/'+'/train', graph=sess.graph)
		valid_writer = tf.train.SummaryWriter('tmp/'+'/valid')

		tf.initialize_all_variables().run()
		print("Initialized")
		model.set_kp_value('fc1/dropout', 0.5)
		model.set_kp_value('fc2/dropout', 0.5)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(training_steps):
			t = time.time()
			offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
			batch_X = train_dataset[offset:(offset+batch_size), :]
			batch_y = train_labels[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X,
									model.train_y : batch_y})
			_, tloss, tacc, tmrg_summ = sess.run([optimizer, train_loss, train_accuracy, \
										merged_summary], feed_dict=train_feed_dict)
			train_losses[step], train_acc[step] = tloss, tacc
			train_writer.add_summary(tmrg_summ, step)
			lr = learning_rate.eval()
			print('Epoch: %d\tLoss: %.4d\tTrain Acc:%.2f%%\tTime Cost: %d\tLearning Rate: %.4f'\
				%(step, tloss, (tacc*100), (time.time()-t), lr))
			if step%100==0:
				valid_feed_dict = dict(model.kp_reference_feed_dict)
				valid_feed_dict.update({model.train_X : batch_X,
							  	model.train_y : batch_y})
				vacc, vloss, vmrg_summ = sess.run([valid_accuracy, valid_loss, merged_summary], \
									 	feed_dict=valid_feed_dict)
				valid_losses.append(vloss)
				valid_acc.append(vacc)
				valid_writer.add_summary(vmrg_summ, step)
				print('Epoch: %d\tLoss: %.4f\tTrain Acc: %.2f%%\tTime Cost: %d\tLearning Rate: %.4f\tValid Acc: %.2f%%' \
                 %(step, tloss, (tacc*100),(time.time()-t), lr,  (vacc*100)))
		print("Finished training")
		tacc = sess.run(test_accuracy, feed_dict=model.kp_reference_feed_dict)
		print("Test accuracy: %.2f%%" %(tacc*100))
	# prepare data needs to be saved
	hyperparams = {'numOfConvFilter' : [4, 4, 4], 'numOfFCNeuron': [512, 256, 10], 
					'init_lr': 0.01, 'augmentaion':True, 'decay_rate': 0.95, 
					'decay_step': 200, 'keep_prob': [0.5, 0.5], 'epoches': training_steps}
	training_data = {'train_losses' : train_losses, 'train_acc' : train_acc, \
					 'valid_losses' : valid_losses, 'valid_acc' : valid_acc, \
					 'test_acc' : tacc, 'hyperparams' : hyperparams}
	return training_data

if __name__=='__main__':
	training_steps = raw_input("How many traing steps?")
	training_data = convnet_model(int(training_steps))
	save_data_name = 'train_data0.1'
	with open(save_data_name, 'wb') as fh:
		pickle.dump(training_data, fh, protocol=2)


