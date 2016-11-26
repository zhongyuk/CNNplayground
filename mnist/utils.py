import numpy as np
import pandas as pd

def load_data(data_filename):
	data = pd.read_csv(data_filename)
	if data.shape[1]==785: # training set
		y = data[['label']]
		X = data[data.columns[1:]]
	else: # testing set
		y = None
		X = data
	X = X.as_matrix().astype(np.float32)
	if y is not None:
		y = y.as_matrix().astype(np.float32)
		y = np.arange(10)==y[:, None]
		y = y.astype(np.float32)
		y = np.squeeze(y, axis=1)
	return X, y

def reshape_data(X):
	return np.reshape(X, [-1, 28, 28, 1])


def make_submission(pred, filename):
	pred_dict = {'ImageId' : range(1, pred.shape[0]+1),
				 'Label'   : pred}
	pred_df = pd.DataFrame(pred_dict)
	pred_df.to_csv(filename, index=False)