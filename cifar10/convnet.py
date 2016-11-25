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
	dataset_list = prepare_cifar10_input(data_dir)
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
	conv_layers = {'conv1': 3, 'conv2': 5, 'conv3': 3 }

	for layer_name, filter_size in conv_layers.items():
		model.add_conv2d_layer(layer_name, filter_size, conv_depth,  conv_wt_initializer)
		model.add_batchnorm_layer(layer_name+"/batchnorm")
		model.add_act_layer(layer_name+"/activation")
		model.add_pool_layer(layer_name+"/pool")

	fc_wt_initializer = tf.contrib.layers.variance_scaling_initializer()
	model.add_fc_layer("fc1", 256, fc_wt_initializer)
	model.add_batchnorm_layer("fc1/batchnorm")
	model.add_act_layer("fc1/activation")
	model.add_dropout_layer("fc1/dropout")

	model.add_fc_layer("fc2", 10, conv_wt_initializer)
	model.add_batchnorm_layer("fc2/batchnorm")
	model.add_act_layer("fc2/activation")

	model.setup_learning_rate(0.001, exp_decay=True, decay_steps=200, \
							 decay_rate=0.95, staircase=False,)

	train_loss = model.compute_train_loss(add_output_summary=False)
	valid_loss = model.compute_valid_loss()

	train_accuracy = model.evaluation("train")
	valid_accuracy = model.evaluation("valid")
	test_accuracy = model.evaluation("test")

	l2_reg_factor = 0.1
	optimizer = model.setup_optimizer(tf.train.AdamOptimizer, l2_reg=True, l2_reg_factor=l2_reg_factor)
	merged_summary = model.merge_summaries()
	graph = model.get_graph()

	# variables need to be saved
	train_losses, valid_losses = np.zeros(training_steps), np.zeros(training_steps)
	train_acc, valid_acc = np.zeros(training_steps), np.zeros(training_steps)

	with tf.Session(graph=graph) as sess:
		print("Create summary writers")
		train_writer = tf.train.SummaryWriter('tmp/'+'/train', graph=sess.graph)
		valid_writer = tf.train.SummaryWriter('tmp/'+'/valid')

		tf.initialize_all_variables().run()
		print("Initialized")
		for step in range(training_steps):
			t = time.time()
			offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
			batch_X = train_dataset[offset:(offset+batch_size), :]
			batch_y = train_labels[offset:(offset+batch_size), :]
			feed_dict = {model.train_X : batch_X,
						model.train_y : batch_y,
						model.keep_prob : 0.3}
			_, tloss, tacc, tmrg_summ = sess.run([optimizer, train_loss, train_accuracy, \
										merged_summary], feed_dict=feed_dict)
			train_losses[step], train_acc[step] = tloss, tacc
			train_writer.add_summary(tmrg_summ, step)
			feed_dict[model.keep_prob] =  1.0
			vacc, vloss, vmrg_summ = sess.run([valid_accuracy, valid_loss, merged_summary], \
									 feed_dict=feed_dict)
			valid_losses[step], valid_acc[step] = vloss, vacc
			valid_writer.add_summary(vmrg_summ, step)
			print('Epoch: %d:\tLoss: %f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%\tTime Cost: %.1f' \
                 %(step, tloss, (tacc*100), (vacc*100), (time.time()-t)))
		print("Finished training")
		tacc = sess.run(test_accuracy, feed_dict={model.keep_prob : 1.0})
		print("Test accuracy: %.2f%%" %(tacc*100))
	training_data = {'train_losses' : train_losses, 'train_acc' : train_acc, \
					 'valid_losses' : valid_losses, 'valid_acc' : valid_acc, \
					 'test_acc' : tacc}
	return training_data

if __name__=='__main__':
	training_steps = raw_input("How many traing steps?")
	training_data = convnet_model(int(training_steps))


