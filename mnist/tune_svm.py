import numpy as np
from utils import *
from six.moves import cPickle as pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def tune_svm(train_filename):
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

if __name__=='__main__':
	train_filename = "/Users/Zhongyu/Documents/projects/kaggle/mnist/train.csv"
	tune_svm(train_filename)