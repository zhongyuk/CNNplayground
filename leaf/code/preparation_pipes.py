from utils import *
from sklearn.model_selection import train_test_split
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground")
from preprocess import center_data

def simple_MLP_pipe(train_filename, test_size, center=True):
	"""
	parameters:
		train_fileanme: train csv data filename
		test_size: 	floating point or integer
	return: ready to feed into deep learning model
		train_X: np.ndarray np.float32
		train_y: one hot encoded np.ndarray np.float32
		test_X : np.ndarray np.float32
		test_y : one hot encoded np.ndarray np.float32
	"""
	# load data
	X, y, species_dict = load_extract_feature(train_filename, True)
	# train test split
	train_X, test_X, train_y, test_y = train_test_split(X, y, 
		test_size=test_size, stratify=y, random_state=924)
	# one hot encode all ys
	train_y = one_hot_encode(train_y)
	test_y  = one_hot_encode(test_y)
	# simple preprocess Xs
	train_X = train_X.as_matrix().astype(np.float32)
	test_X  = test_X.as_matrix().astype(np.float32)
	if center:
		train_X = center_data(train_X, axis=0)
		test_X  = center_data(test_X, axis=0)
	# print out train and test set sizes
	print("training set size: \t%s\t%s " %(train_X.shape, train_y.shape))
	print("testing set size: \t%s\t%s " %(test_X.shape, test_y.shape))
	return train_X, train_y, test_X, test_y

def pca_MLP_pipe(train_filename, test_size):
	"""
	parameters:
		train_fileanme: train csv data filename
		test_size: 	floating point or integer
	return: ready to feed into deep learning model
		train_X: np.ndarray np.float32
		train_y: one hot encoded np.ndarray np.float32
		test_X : np.ndarray np.float32
		test_y : one hot encoded np.ndarray np.float32
	"""
	# load_data
	X, y, species_dict = load_extract_feature(train_filename, True)
	# train test split
	train_X, test_X, train_y, test_y = train_test_split(X, y, 
		test_size=test_size, stratify=y, random_state=924)
	# one hot encode all ys
	train_y = one_hot_encode(train_y)
	test_y  = one_hot_encode(test_y)
	# PCA process Xs
	feature_names = ['margin', 'texture', 'shape']
	num_component = [8, 10, 3]
	train_list, test_list = [], []
	for feature, n in zip(feature_names, num_component):
		train_transformed, test_transformed = pca_transform(train_X, test_X, feature, n)
		train_list.append(train_transformed)
		test_list.append(test_transformed)
	train_X = np.concatenate(train_list, axis=1).astype(np.float32)
	test_X = np.concatenate(test_list, axis=1).astype(np.float32)
	# print out train and test set sizes
	print("training set size: \t%s\t%s " %(train_X.shape, train_y.shape))
	print("testing set size: \t%s\t%s " %(test_X.shape, test_y.shape))
	return train_X, train_y, test_X, test_y

def CNN_pipe(train_filename, test_size, order='C', center=True, whiten=None):
	"""
	parameters:
		train_fileanme: train csv data filename
		test_size: 	floating point or integer
	return: ready to feed into deep learning model
		train_X: np.ndarray np.float32
		train_y: one hot encoded np.ndarray np.float32
		test_X : np.ndarray np.float32
		test_y : one hot encoded np.ndarray np.float32
	"""
	# load_data
	X, y, species_dict = load_extract_feature(train_filename, True)
	# train test split
	train_X, test_X, train_y, test_y = train_test_split(X, y, 
		test_size=test_size, stratify=y, random_state=924)
	# one hot encode all ys
	train_y = one_hot_encode(train_y)
	test_y  = one_hot_encode(test_y)
	# preprocess Xs
	feature_names = ['margin', 'texture', 'shape']
	train_list, test_list = [], []
	for feature in feature_names:
		train_reshaped = shape_feature3D(train_X, feature, order, center, whiten)
		test_reshaped = shape_feature3D(test_X, feature, order, center, whiten)
		train_list.append(train_reshaped[:,:,:,None])
		test_list.append(test_reshaped[:,:,:,None])
	train_X = np.concatenate(train_list, axis=3).astype(np.float32)
	test_X = np.concatenate(test_list, axis=3).astype(np.float32)
	# print out train and test set sizes
	print("training set size: \t%s\t%s " %(train_X.shape, train_y.shape))
	print("testing set size: \t%s\t%s " %(test_X.shape, test_y.shape))
	return train_X, train_y, test_X, test_y

if __name__=='__main__':
	train_filename = '../train.csv'
	train_X, train_y, test_X, test_y = simple_MLP_pipe(train_filename, 330)
	train_X, train_y, test_X, test_y = pca_MLP_pipe(train_filename, 330)
	train_X, train_y, test_X, test_y = CNN_pipe(train_filename, 330)

