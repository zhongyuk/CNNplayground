## Convolutional Neural Network Playground 

- The **aim** of this repo is for a hands-on exploring and parcticing deep learning convolutional neural network upon various dataset. 

#### Programming language
- Python 2.7 for master branch
- Python 3.5 for python35 branch

#### Library Dependencies
- master branch: 
	- TensorFlow r0.11
	- numpy 1.11.2
	- scipy 0.18.1
	- scikit-learn 0.18.1
	- matplotlib 1.5.3
	- pandas 0.19.0
	- jupyter 1.0.0
- python35 branch
	- TensorFlow r0.12
	- numpy 1.11.0b1
	- scipy 0.18.1
	- scikit-learn 0.18.1
	- matplotlib 1.5.3
	- pandas 0.19.1
	- jupyter 1.0.0

#### Files Explained
- **/capstone directory** contains the code, notebooks, data, and report for the Udacity Machine Learning Nanodegree capstone project.

- **cifar10 directory** stores the code and files for further exploring the CIFAR10 dataset using deep learning techniques after the Udacity Machine Learning Nanodegree capstone project.

- **mnist directory** is a project for building a digit recoginizer upon MNIST dataset (incorporating kaggle playgorund competition) using machine learning and deep learning techniques.

- **studybn directory** contains a few scripts for studying and testing bach normalization and a notebook for demonstrating how to properly use TensorFlow `tf.contrib.layers.batch_norm`. 

- **cnn.py and test_cnn.py**: the former is a quick wrapper building on top of TensorFlow to enable an easier and faster usage for building a CNN model (note that only the commonly recommended/default model building options are included and implemented in `cnn.py`); the latter performs unit testing on the interfaces provided by `cnn.py`.