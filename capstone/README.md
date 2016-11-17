## Image Classification using Convolutional Neural Network 
*Udacity Machine Learning Nanodegree Capstone Project *

#### Dataset
[CIFAR-10](http://www.cs.utoronto.ca/~kriz/cifar.html)

#### Programming Language
Python 2.7

#### Software Dependencies
- TensorFlow
- Numpy
- scikit-learn
- matplotlib

#### Files Explained
- **In /code directory**
All the development codes for this project are in /code. For the .py files which end with a numerical number (0~4): the convnet_devp0.py is the earliest development code, convnet_devp4.py is the latest development code. Most of the training data files stored in /train_data are results collected by running convnet_devp2.py with different hyperparameter settings. The prepare_input.py alone file contains the functions for lodaing, preprocessing input data. 

- **In /notebook directory**
All the jupyter notebooks end with a numerical number (0~3). The number corresponds to a project development stage (which matches with the numerical number in /code/convnet_devp*.py). Each notebook records from coarse to fine hyperparameter tunings and plots/analyzes some training data stored in /train_data.
	- experiment*.ipynb includes coarse to fine hyper-parameter tuning and analysis of training_data records
	- train_record*.ipynb only contains analysis of training_data records

- **In /train_data directory**
Training data is recorded for different CNN models and stored in pickle files. The first numerical number in each pickle data file indicates the corresponding stage of the development code (training_data_stack2.* is collected by running convnet_devp2). The second numerical number correspondes to different hyperparameter settings. The details pertaining to the hyperparameter settings can be found in the corresponding notebook (/notebook/experiment2.ipynb)

- **report.pdf**
The project report for Udacity Machine Learning Nanodegree capstone project. It summaries the work done from (/code) convnet_devp0.py to convnet_devp2.py, (/notebook) experiment0.ipynb to experiment2.ipynb, and the training results from (/train_data) training_data_stack0. to training_data_stack2.


