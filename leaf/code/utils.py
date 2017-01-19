import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys
sys.path.append("/Users/Zhongyu/Documents/projects/CNNplayground")
from preprocess import center_data, pca_whiten, zca_whiten
import PIL
from PIL import Image
from six.moves import cPickle as pickle
import time
from os import listdir

def load_extract_feature(filename, is_training):
	"""
	functionalities:
		1. Load extracted features csv data
		2. Numerically encode the target/label class - 'species'
	parameters:
		filename: csv filename
		is_training: if True, load the train.csv and return y and species_dict along with X
					 if False, load the test.csv and return X
	return:
		X: a data frame of extracted features: margin, texture, shape (unique sample ID is not included)
		y: a data series of numerically encoded species
		species_dict: a dictionary mapping each specie to its unique numerical encode ID
	"""
	data = pd.read_csv(filename)
	if is_training:
		species_dict = load_species_encode()
		data = data.replace({'species':species_dict})
		X = data.drop(['id','species'], axis=1)
		y = data['species']
		return X, y, species_dict
	else:
		X = data.drop(['id'], axis=1)
		return X

def create_species_encode():
	"""
	create and save species dict for convenience
	"""
	data = pd.read_csv("../train.csv")
	species = sorted(data.species.unique())
	species_dict = {species: index for index, species in enumerate(species)}
	return species_dict

def one_hot_encode(y):
	"""
	functionalities: One hot encode y and change data type to float32
	parameters: 
		y: a panda data series or numpy 1d array
	return:
		y_ohe: a 2D numpy ndarray with dtype=np.float32
	"""
	num_class = y.unique().shape[0]
	y_ohe = np.arange(num_class)==np.array(y)[:, None]
	assert(y_ohe.shape==(y.shape[0], num_class))
	return y_ohe.astype(np.float32)

def pca_transform(train_X, test_X, feature, n_components, whiten=True):
	"""
	parameters:
		train_X: training set data frame 
		test_X: testing set data frame
		feature: a string indicating which columns in the DF to perform PCA on
		n_components: number of principle components to keep
		whiten: whiten the data during PCA transformation or not
	return: 
		train_feat_transform: PCA tranformed training features np.ndarray
		test_feat_transform: PCA tranformed testing features np.ndarray
	"""
	cols = [col for col in list(train_X.columns) if feature in col]
	train_features, test_features = train_X[cols].as_matrix(), test_X[cols].as_matrix()
	train_features, test_features = center_data(train_features), center_data(test_features)
	pca_obj = PCA(n_components=n_components, whiten=whiten)
	pca_obj.fit(train_features)
	print("total explained variance by %d principle components: \t %.4f" \
		%(n_components, sum(pca_obj.explained_variance_ratio_)))
	train_feat_transform = pca_obj.transform(train_features)
	test_feat_transform = pca_obj.transform(test_features)
	assert(train_feat_transform.shape==(train_X.shape[0], n_components))
	assert(test_feat_transform.shape==(test_X.shape[0], n_components))
	return train_feat_transform, test_feat_transform

def shape_feature3D(X, feature, order='C', center=True, whiten=None):
	"""
	parameters:
		X: feature data frame
		feature: a string uses to select features
		order: 'C' or 'F'
		center: if True: center data; if False: not center data
		whiten: if None: not whiten; if 'pca': pca_whiten; if 'zca': zca_whiten
	return: 
		feat3D: reshaped features in 3D np.ndarry
	"""
	if order not in ('C', 'F'):
		msg = order+" is not a valid argument"
		raise ValueError(msg)
	if whiten not in (None, 'pca', 'zca'):
		msg = whiten+" is not a valid argument"
		raise ValueError(msg)
	cols = [col for col in list(X.columns) if feature in col]
	features = X[cols].as_matrix()
	if center:
		features = center_data(features)
	if whiten=='pca':
		features = pca_whiten(features)
	elif whiten=='zac':
		features = zca_whiten(features)
	feat3D = features.reshape((X.shape[0], 8, 8), order=order)
	return feat3D

def downsize_image(img, target_length=64):
	"""
	functionalities: downsize an image to a 64 by 64 image while preserving aspect ratio
	parameters:
		img: a PIL.JpegImagePlugin.JpegImageFile, img.size is expected to be > 64
		target_height: target height after downsizing
	return:
		img_downsized: a 64 by 64 numpy ndarray
	"""
	h, w = img.size[0], img.size[1]
	ratio = float(h)/w
	if h>w:
		new_size = (target_length, int(round(target_length/ratio)))
	else:
		new_size = (int(round(target_length*ratio)), target_length)
	img_downsized = img.resize(new_size, PIL.Image.ANTIALIAS)
	img_np = np.array(img_downsized)
	img_square = np.zeros([target_length, target_length],dtype=img_np.dtype)
	img_square[:img_np.shape[0], :img_np.shape[1]] = img_np
	return img_square

def batch_downsize(image_dir, target_length=64):
	"""
	functionalities: 
		Batch downsize all images and save into a dictionary using 
		sample ID as key and downsized square image numpy array as value
	parameters:
		image_dir: the directory path to all images
		target_length: target size length after downsizing images
	returns:
		img_dict: dictonary for storing all downsized square np image array
	"""
	img_dict = {}
	all_image_filenames = listdir(image_dir)
	t = time.time()
	for img_fn in all_image_filenames:
		img_id = int(img_fn.split('.')[0])
		with Image.open(image_dir+img_fn) as img:
			img_np = downsize_image(img, target_length=target_length)
			assert(img_np.shape==(target_length, target_length))
			img_dict[img_id] = img_np
	assert(len(img_dict)==1584)
	tcost = time.time()-t
	print("the time cost for downsizing all %d images: %.2f seconds" %(len(img_dict), tcost))
	return img_dict

def pickle_data(data, filename):
	"""
	functionaly: pickle/save data
	parameters:
		data: python data 
		filename: path+filename for saving data
	return:
		None
	"""
	with open(filename, "wb") as fh:
		pickle.dump(data, fh, protocol=2)

def unpickle_data(filename):
	"""
	functionaly: unpickle/load filename
	parameters:
		filename: path+filename for loading data
	return:
		data: loaded data
	"""
	with open(filename, 'rb') as fo:
		data = pickle.load(fo)
	return data



if __name__=='__main__':
	# save species encoder dictionary
	species_encode = create_species_encode() 
	pickle_data(species_encode, "../species_encode")
	# load species numeric encoder
	species_dict = unpickle_data("../species_encode")
	assert(len(species_dict)==99)

	#save batch downsized images
	image_dir = '/Users/Zhongyu/Documents/projects/CNNplayground/leaf/images/'
	img_dict = batch_downsize(image_dir, 64)
	pickle_data(img_dict, "../images_downsized")
	# load downsized image data
	t = time.time()
	img_data = unpickle_data("../images_downsized")
	tcost = time.time()-t
	print("time costs for loading saved image data: %.2f seconds" %(tcost))


	 
