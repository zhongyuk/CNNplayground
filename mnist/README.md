## Handwritten Digit Recognizer - MNIST dataset
**Incorporating with [Kaggle's playground competition](https://www.kaggle.com/competitions) - the Digit Recognizer, this project aims at hands-on practicing and exploring various convolution neural network techniques**

#### Dataset
[MNIST]
- [kaggle channel](https://www.kaggle.com/c/digit-recognizer/data)
- [MNIST database channel](http://yann.lecun.com/exdb/mnist/)

#### Programming Language
Python 2.7

#### Library Dependencies
- TensorFlow r0.11
- numpy 1.11.2
- scipy 0.18.1
- scikit-learn 0.18.1
- pandas 0.19.0
- jupyter 1.0.0

#### Files Explained
- **utils.py** contains a few utility functions for loading the MNIST data from *.csv files, preprocessing data, reshaping data for CNN input, saving data, making kaggle competition submission files, etc.

- **models.py** contains a few models for undertaking the digit recognizer job: including a shallow simple fully connected neural network, several convolutional neural networks which different architecture and parameter settings, and a simple support vector machine model (using scikit-learn framework). Additionally, it contains a `train_model` function to initiate training any single model and a `compare_models` function for comparing performance of two models: `cnn_c2f2` (a convolutional neural network with 2 <3x3> convolutional layers and 2 fully connected layers) and `cnn_c2f2nin`(a convolutional neural network with 1 <3x3> convolutional layer and 1 <1x1> convolutional layer and 2 fully connected layers - a hands-on exploration of the [Network In Network](https://arxiv.org/pdf/1312.4400v3.pdf) reference).

- **tune_svm.py** implements a grid search fine-tuning over different hyper-parameter settings of a Support Vector Machine (SVM) model. As it turns out that SVM model on this dataset with regular grid search hyper-parameter tuning appears to be *very* computational expensive.

- **fractmaxpool.py** is an hands-on exploration using the MNIST dataset on the [fractional max pooling technique](https://arxiv.org/pdf/1412.6071v4.pdf). The practice model was built upon regular convolutional neural network instead of [spatially-sparse convolutional neural network](https://arxiv.org/pdf/1409.6070v1.pdf). A coarse exploration shows that CNN with farctial max pooling has potential in rendering good performances, however due to the "stochastic" in the pooling process, the performance also shows some "randomness", hence setting `seed` argument is recommended for reproducibility. Moreover, fractial max pooling appears to work well with deeper CNN and smaller filter size. 

- **convnet.py** is a simple CNN model built through using the `cnn.py` interface with TensorBoard visulization turned on.

- **ensemble.py** fulfills two ensemble methods: bagging and stacking. Explicitly it implements a `kfold` function for performing the first stage of stacking ensemble technique and a `stacking` function for performing the second stage. The `bagging` function takes in the `kfold` trained multiple model results and performs a simple majority vote to reach the final prediction.

- **/kfold_data directory** stores the results obtained by k-fold training multiple models (models loaded from `models.py`)