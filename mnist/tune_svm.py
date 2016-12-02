import numpy as np
from utils import *
from six.moves import cPickle as pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import time

def tune_svm(train_filename):
	#GridSearchCV never stops running... 
	#Consider code own SVM or manual tune SVM...
	X, y = load_data(train_filename)
	label = np.argmax(y, 1)
	params = {'C'	 	: [1e-1, 1., 1e1],
			  'kernel'	: ['rbf', 'sigmoid'],
			  'gamma'	: ['auto', 1e-2, 1e-1, 1., 1e1]}
	print "Start tunning..."
	clf_obj = SVC()
	gscv_obj = GridSearchCV(estimator=clf_obj, param_grid=params)
	gscv_obj.fit(X, label)
	best_param = gscv_obj.best_params_
	best_score = gscv_obj.best_score_
	print "The best parameter setting is..."
	print best_param
	print "The achieved best accuracy score is..."
	print best_score
	result = gscv_obj.cv_results_
	with open("GridSrchResult", "w") as fh:
		pickle.dump(result, fh)
	print "Done tunning SVM."

def manual_tune_svm(train_filename, k, C, gamma):
	X, y = load_data(train_filename)
	label = np.argmax(y, 1)
	skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=5261)
	skf_iter_obj = iter(skf.split(X, label))
	for i in range(k):
		print("="*12, "training fold ", i, "="*12)
		train_index, test_index = skf_iter_obj.next()
		train_X, test_X = X[train_index, :], X[test_index, :]
		train_y, test_y = label[train_index], label[test_index]
		t = time.time()
		svm_model(train_X, train_y, test_X, test_y, C, gamma)
		print "Time cost for training 1 fold: ", time.time()-t


def svm_model(train_X, train_y, test_X, test_y, C, gamma):
	"""
	A Support Vector Machine Model
	"""
	clf_obj = SVC(C=C, gamma=gamma)
	clf_obj.fit(train_X, train_y)
	train_accuracy = clf_obj.score(train_X, train_y)
	valid_accuracy = clf_obj.score(test_X, test_y)
	print('Train Acc: %.2f%%\tValid Acc: %.2f%%') \
		 %((train_accuracy*100), (valid_accuracy*100))
	print("Finished training")


if __name__=='__main__':
	train_filename = "/Users/Zhongyu/Documents/projects/kaggle/mnist/train.csv"
	#tune_svm(train_filename)
	manual_tune_svm(train_filename, k=2, C=7, gamma='auto')
