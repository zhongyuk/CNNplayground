import sys
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground/")
from cnn import *

def train_model(train_filename, test_filename, training_steps):
	X, y = load_data(train_filename)
	kX, _ = load_data(test_filename)
	train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=2000,
										 random_state=263, stratify=y)
	valid_X, test_X, valid_y, test_y = train_test_split(valid_X, valid_y, test_size=1000,
										 random_state=932, stratify=valid_y)

	train_X = reshape_data(train_X)
	valid_X = reshape_data(valid_X)
	test_X = reshape_data(test_X)
	kaggle_X = reshape_data(kX)
	print train_X.shape, train_y.shape
	print valid_X.shape, valid_y.shape
	print test_X.shape, test_y.shape

	batch_size = 64
	input_shape = [batch_size, 28, 28, 1]
	conv_depth = 4
	num_class = 10

	# Build a convnet graph
	model = cnn_graph(input_shape, num_class)
	model.setup_data(batch_size, test_X, test_y, valid_X, valid_y)
	conv_wt_initializer = tf.truncated_normal_initializer(stddev=.10)
	conv_layers = [('conv1', 3), ('conv2', 3)]

	for conv_layer in conv_layers:
		layer_name, filter_size = conv_layer[0], conv_layer[1]
		model.add_conv2d_layer(layer_name, filter_size, conv_depth,  conv_wt_initializer)
		model.add_batchnorm_layer(layer_name+"/batchnorm")
		model.add_act_layer(layer_name+"/activation")
		model.add_pool_layer(layer_name+"/pool")

	fc_wt_initializer = tf.contrib.layers.variance_scaling_initializer()
	model.add_fc_layer("fc1", 1024, fc_wt_initializer)
	model.add_batchnorm_layer("fc1/batchnorm")
	model.add_act_layer("fc1/activation")
	model.add_dropout_layer("fc1/dropout")

	model.add_fc_layer("fc2", 256, fc_wt_initializer)
	model.add_batchnorm_layer("fc2/batchnorm")
	model.add_act_layer("fc2/activation")
	model.add_dropout_layer("fc2/dropout")

	model.add_fc_layer("fc3", 10, conv_wt_initializer)
	model.add_batchnorm_layer("fc3/batchnorm")
	model.add_act_layer("fc3/activation")

	model.setup_learning_rate(0.01, exp_decay=True, decay_steps=200, \
							 decay_rate=0.85, staircase=False)

	train_loss = model.compute_train_loss(add_output_summary=False)

	train_accuracy = model.evaluation("train")
	valid_accuracy = model.evaluation("valid")
	test_accuracy  = model.evaluation("test")

	kaggle_y = model.make_prediction(kaggle_X)

	#l2_reg_factor = 0.01
	optimizer = model.setup_optimizer(tf.train.AdamOptimizer)
	#optimizer = model.setup_optimizer(tf.train.AdamOptimizer, l2_reg=True, l2_reg_factor=l2_reg_factor)
	merged_summary = model.merge_summaries()
	graph = model.get_graph()

	with tf.Session(graph=graph) as sess:
		print("Create summary writers")
		train_writer = tf.train.SummaryWriter('tmp/'+'/train', graph=sess.graph)
		valid_writer = tf.train.SummaryWriter('tmp/'+'/valid')
		tf.initialize_all_variables().run()
		print("Initialized")
		model.set_kp_value('fc1/dropout', 0.5)
		model.set_kp_value('fc2/dropout', 0.6)
		train_feed_dict = model.get_kp_collection_dict()
		for step in range(training_steps):
			offset = (step*batch_size)%(train_X.shape[0]-batch_size)
			batch_X = train_X[offset:(offset+batch_size), :]
			batch_y = train_y[offset:(offset+batch_size), :]
			train_feed_dict.update({model.train_X : batch_X,
									model.train_y : batch_y})
			_, tloss, tacc, tmrg_summ = sess.run([optimizer, train_loss, train_accuracy, \
										merged_summary], feed_dict=train_feed_dict)
			train_writer.add_summary(tmrg_summ, step)
			valid_feed_dict = dict(model.kp_reference_feed_dict)
			valid_feed_dict.update({model.train_X : batch_X,
							 	    model.train_y : batch_y})
			if step%50==0:
				vacc, vmrg_summ = sess.run([valid_accuracy, merged_summary], \
								feed_dict=valid_feed_dict)
				valid_writer.add_summary(vmrg_summ, step)
				print('Epoch: %d\tLoss: %f\tTrain Acc: %.2f%%\tValid Acc: %.2f%%' \
                 	%(step, tloss, (tacc*100), (vacc*100)))
		print("Finished training")
		tacc = sess.run(test_accuracy, feed_dict=model.kp_reference_feed_dict)
		print("Test accuracy: %.2f%%" %(tacc*100))
		print("Making prediction.")
		kaggle_pred = sess.run(kaggle_y, feed_dict=model.kp_reference_feed_dict)
		print("Done making predictions.")
	return kaggle_pred

if __name__=="__main__":
	train_filename = "/Users/Zhongyu/Documents/projects/kaggle/mnist/train.csv"
	test_filename = "/Users/Zhongyu/Documents/projects/kaggle/mnist/test.csv"
	training_steps = raw_input("How many traing steps?")
	predictions = train_model(train_filename, test_filename, int(training_steps))
	make_submission(predictions, "pred3.csv")