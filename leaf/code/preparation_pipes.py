from utils import *
from sklearn.model_selection import train_test_split
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground")
from preprocess import center_data, zca_whiten, pca_whiten, augment_data

def simple_MLP_pipe(train_filename, test_size, center=True, random_state=None):
	"""
	parameters:
		train_fileanme: train csv data filename
		test_size: 	floating point or integer
	return: ready to feed into deep learning model
		train_X: np.ndarray np.float32 shape=(_, 192)
		train_y: one hot encoded np.ndarray np.float32 shape=(_, 99)
		test_X : np.ndarray np.float32 shape=(_, 192)
		test_y : one hot encoded np.ndarray np.float32 shape=(_, 99)
	"""
	# load data
	X, y = load_extract_feature(train_filename, True)
	# train test split
	train_X, test_X, train_y, test_y = train_test_split(X, y, 
		test_size=test_size, stratify=y, random_state=random_state)
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

def pca_MLP_pipe(train_filename, test_size, random_state=None):
	"""
	parameters:
		train_fileanme: train csv data filename
		test_size: 	floating point or integer
	return: ready to feed into deep learning model
		train_X: np.ndarray np.float32 shape=(_, 21)
		train_y: one hot encoded np.ndarray np.float32 shape=(_, 99)
		test_X : np.ndarray np.float32 shape=(_, 21)
		test_y : one hot encoded np.ndarray np.float32 shape=(_, 99)
	"""
	# load_data
	X, y = load_extract_feature(train_filename, True)
	# train test split
	train_X, test_X, train_y, test_y = train_test_split(X, y, 
		test_size=test_size, stratify=y, random_state=random_state)
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

def CNN_pipe_csv(train_filename, test_size, order='C', center=True, 
				 whiten=None, random_state=None):
	"""
	parameters:
		train_fileanme: train csv data filename
		test_size: 	floating point or integer
		center: center the data or not
		whiten: None - not whitening input, 'pca' - pca whitening, 'zca' - zca whitening
	return: ready to feed into deep learning model
		train_X: np.ndarray np.float32 shape=(_, 8, 8, 3)
		train_y: one hot encoded np.ndarray np.float32 shape=(_, 99)
		test_X : np.ndarray np.float32 shape=(_, 8, 8, 3)
		test_y : one hot encoded np.ndarray np.float32 shape=(_, 99)
	"""
	# load data
	X, y = load_extract_feature(train_filename, True)
	# train test split
	train_X, test_X, train_y, test_y = train_test_split(X, y, 
		test_size=test_size, stratify=y, random_state=random_state)
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

def CNN_pipe_img(train_filename, test_size, whiten=None, random_state=None):
	"""
	parameters:
		train_fileanme: train csv data filename
		test_size: 	floating point or integer
		whiten: None - not whitening input, 'pca' - pca whitening, 'zca' - zca whitening
	return: ready to feed into deep learning model
		train_X: np.ndarray np.float32 shape=(_, 32, 32, 1)
		train_y: one hot encoded np.ndarray np.float32 shape=(_, 99)
		test_X : np.ndarray np.float32 shape=(_, 32, 32, 1)
		test_y : one hot encoded np.ndarray np.float32 shape=(_, 99)
	"""
	# load data
	X, y = load_image_data(train_filename, True)
	# train test split
	train_X, test_X, train_y, test_y = train_test_split(X, y, 
		test_size=test_size, stratify=y, random_state=random_state)
	# one hot encode all ys
	train_y = one_hot_encode(train_y)
	test_y  = one_hot_encode(test_y)
	# preprocess Xs
	train_X = center_whiten_image(train_X, whiten=whiten)
	test_X = center_whiten_image(test_X, whiten=whiten)
	train_X = train_X[:,:,:,None].astype(np.float32)
	test_X = test_X[:,:,:,None].astype(np.float32)
	#train_X, train_y = augment_data(train_X, train_y)
	# print out train and test set sizes
	print("training set size: \t%s\t%s " %(train_X.shape, train_y.shape))
	print("testing set size: \t%s\t%s " %(test_X.shape, test_y.shape))
	return train_X, train_y, test_X, test_y

def combined_model_pipe(train_filename, test_size, random_state=None, whiten=None):
	"""
	parameters:
		train_fileanme: train csv data filename
		test_size: 	floating point or integer
		whiten: None - not whitening input, 'pca' - pca whitening, 'zca' - zca whitening
		random_state: random state integer for sklearn.model_selection.train_test_split
					  seed for train_test_split must be between 0 and 4294967295 (inclusive)
	return: ready to feed into deep learning model
		train_X_csv: np.ndarray np.float32 shape=(_, 21) or (_, 192)
		train_X_img: np.ndarray np.float32 shape=(_, 32, 32, 1)
		train_y: one hot encoded np.ndarray np.float32 shape=(_, 99)
		test_X_csv : np.ndarray np.float32 shape=(_, 21) or (_, 192)
		test_X_img: np.ndarray np.float32 shape=(_, 32, 32, 1)
		test_y : one hot encoded np.ndarray np.float32 shape=(_, 99)
	"""
	if random_state==None:
		random_state = np.random.randint(low=0, high=4294967296)
	train_X_csv, train_y_csv, test_X_csv, test_y_csv = pca_MLP_pipe(train_filename, test_size,
																 random_state=0)
	train_X_img, train_y_img, test_X_img, test_y_img = CNN_pipe_img(train_filename, test_size,
																 random_state=0)
	assert(np.array_equal(train_y_csv, train_y_img))
	assert(np.array_equal(test_y_csv,  test_y_img))
	return train_X_csv, train_X_img, train_y_csv, test_X_csv, test_X_img, test_y_csv



if __name__=='__main__':
	train_filename = '../train.csv'
	train_X, train_y, test_X, test_y = simple_MLP_pipe(train_filename, 330)
	train_X, train_y, test_X, test_y = pca_MLP_pipe(train_filename, 330)
	train_X, train_y, test_X, test_y = CNN_pipe_csv(train_filename, 330)
	train_X, train_y, test_X, test_y = CNN_pipe_img(train_filename, 330)
	train_X_csv, train_X_img, train_y, test_X_csv, test_X_img, test_y = combined_model_pipe(train_filename, 110, 924)

