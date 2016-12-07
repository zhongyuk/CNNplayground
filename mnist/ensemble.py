"""
Ensemble Methods: Bagging and Stacking 
train_kfold is a function for performaning the first stage of ensemble stacking model fitting.
train_stack ...
bagging is a ensemble technique for collecting predictions of several predictors and 
performing majority vote to conclude the final predictions.
"""
import numpy as np
from six.moves import cPickle as pickle
from utils import *
from models import *
from sklearn.model_selection import StratifiedKFold
import warnings

def kfold(train_filename, test_filename, k, 
				model, data_filename):
	X, y = load_data(train_filename)
	label = np.argmax(y, 1)
	skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=398745)
	skf_iter_obj = iter(skf.split(X, label))
	corr_labels = None
	predictions = None
	for i in range(k):
		print("="*12, "training fold ", i, "="*12)
		train_index, test_index = skf_iter_obj.next()
		train_X, test_X = X[train_index, :], X[test_index, :]
		train_y, test_y = y[train_index, :], y[test_index, :]
		if corr_labels is None:
			corr_labels = label[test_index]
		else:
			corr_labels = np.concatenate((corr_labels, label[test_index]))
		pred_y = model(train_X, train_y, test_X, test_y)
		if predictions is None:
			predictions = pred_y
		else:
			predictions = np.concatenate((predictions, pred_y))
	print("*"*12, "training whole dataset", "*"*12)
	tX, ty = load_data(test_filename)
	predict_ty = model(X, y, tX, ty)
	# collect data and pickle data
	data = {'labels':corr_labels, 'predict':predictions,
			'test_pred' : predict_ty}
	with open(data_filename, 'wb') as fh:
		pickle.dump(data, fh, protocol=2)
	return data

def train_kfold():
	'''A func to loop through all models and performing kfold training'''
	train_filename = "/Users/Zhongyu/Documents/projects/kaggle/mnist/train.csv"
	test_filename = "/Users/Zhongyu/Documents/projects/kaggle/mnist/test.csv"
	K = 7
	model_names = [snn_f2, cnn_c2f2, cnn_c4f3, svm_model]
	data_filenames = ['snn_f2_kfold', 'cnn_c2f2_kfold', 'cnn_c4f3_kfold','svm_model_kfold']
	for model, data_fn in zip(model_names, data_filenames):
		data = train_kfold(train_filename, test_filename, K, model, data_fn)

def bagging(datafile_list):
    '''Load all predictions trained by different models and perform majority vote'''
    all_test_pred = None
    all_train_pred = None
    for data_file in datafile_list:
        data = unpickle(data_file)
        if all_test_pred is None:
            all_test_pred = np.reshape(data['test_pred'], [-1, 1])
            all_train_pred = np.reshape(data['predict'], [-1, 1])
        else:
            test_pred = np.reshape(data['test_pred'], [-1, 1])
            predict = np.reshape(data['predict'], [-1,1])
            all_test_pred = np.concatenate((all_test_pred, test_pred), axis=1)
            all_train_pred = np.concatenate((all_train_pred, predict), axis=1)
    # majority vote
    train_labels = data['labels']
    train_preds = majority_vote(all_train_pred)
    train_acc = compute_accuracy(train_labels, train_preds)
    print("Bagging ensemble %d models results accuracy score of %.2f%%" \
        %(len(datafile_list), (100*train_acc)))
    test_preds = majority_vote(all_test_pred)
    make_submission(test_preds, 'pred4')
    return test_preds


def majority_vote(data):
	if data.shape[1]%2==0:
		warnings.warn("Even number of columns, will break tie based on the order.")
	return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, data)

def compute_accuracy(label, pred):
	return 1.0*np.sum(label==pred)/pred.shape[0]


if __name__=='__main__':
	prefix = 'kfold_data/'
	filename = ['cnn_c2f2_kfold', 'cnn_c4f3_kfold', 'snn_f2_kfold', 'svm_model_kfold']
	datafile_list = [prefix+fn for fn in filename]
	bagging(datafile_list)
