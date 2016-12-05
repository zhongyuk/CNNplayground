"""
Ensemble Methods: Bagging and Stacking 
train_kfold is a function for performaning the first stage of ensemble stacking model fitting.
train_stack ...
bagging ...
"""
import numpy as np
from six.moves import cPickle as pickle
from utils import *
from models import *
from sklearn.model_selection import StratifiedKFold

def train_kfold(train_filename, test_filename, k, model, 
                training_steps, data_filename):
    X, y = load_data(train_filename)
    label = np.argmax(y, 1)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=398745)
    skf_iter_obj = iter(skf.split(X, label))
    corr_labels = None
    predictions = None
    for i in range(k):
        print("="*12, "training fold ", i, "="*12)
        train_index, test_index = next(skf_iter_obj)
        train_X, test_X = X[train_index, :], X[test_index, :]
        train_y, test_y = y[train_index, :], y[test_index, :]
        if corr_labels is None:
            corr_labels = label[test_index]
        else:
            corr_labels = np.concatenate((corr_labels, label[test_index]))
        pred_y = model(train_X, train_y, test_X, test_y, training_steps)
        if predictions is None:
            predictions = pred_y
        else:
            predictions = np.concatenate((predictions, pred_y))
    print("*"*12, "training whole dataset", "*"*12)
    tX, ty = load_data(test_filename)
    predict_ty = model(X, y, tX, ty, training_steps)
    # collect data and pickle data
    data = {'labels':corr_labels, 'predict':predictions,
            'test_pred' : predict_ty}
    with open(data_filename, 'w') as fh:
        pickle.dump(data, fh)
    return data


if __name__=='__main__':
    train_filename = "/Users/Zhongyu/Documents/projects/CNNplayground/mnist/data/train.csv"
    test_filename = "/Users/Zhongyu/Documents/projects/CNNplayground/mnist/data/test.csv"
    training_steps = 2501
    K = 3
    model_names = [snn_f2, cnn_c2f2, cnn_c4f3, svm_model]
    data_filenames = ['snn_f2_kfold', 'cnn_c2f2_kfold', 'cnn_c4f3_kfold','svm_model_kfold']
    for model, data_fn in zip(model_names, data_filenames):
        data = train_kfold(train_filename, test_filename, K, model, 
                            training_steps, data_fn)
