## Leaf Classification
**Borrowing the [Kaggle playground competition - leaf classification dataset](https://www.kaggle.com/c/leaf-classification), this project aims at exploring and applying deep learning techniques in a slightly more complex data setting (as compared to the MNIST and CIFAR10)**

#### Dataset
- [kaggle leaf dataset](https://www.kaggle.com/c/leaf-classification) includes a set of pre-extracted numeric features stored in .csv files and a set of gray scale JPEG images.

#### Programming Language
Python 2.7

#### Library Dependencies
- TensorFlow r0.11
- numpy 1.11.2
- scipy 0.18.1
- scikit-learn 0.18.1
- pandas 0.19.0
- jupyter 1.0.0
- matplotlib 1.5.3
- Pillow/PIL 4.0.0

#### Files Explained

- **/code dir** stores all the scripts for this project.
	- utils.py contains the very basic building block functions, such as reading in .csv and .jpg data and preprocessing data, etc.
	- preparation_pipes.py utilizes the building blocks from utils.py to accomplish a simple pipeline of work (load in data -> split data into training and testing/validation -> preprocess data -> prepare data <right shape, right format, right precision>). The functions in preparation_pipes.py takes data filename as input and outputs numpy.ndarrays which are ready to feed into deep learning models.
	- extr_feat_models.py stores simple MLP and CNN model templates for classifying leaf classes based on the extracted features (stored in .csv files)

- **/notebooks dir** contains notebooks recording data exploration and model development work. The numeric number indicates the timeline of the notebook. The suffix of "csv" and "image" indicates the focus of the data (the pre-extracted features or the JPEG images)

- images (ignored by git) contains all the JPEG images, each named by an unique sample ID number

- pickle files (ignored by git) include *species_encode* and *images_downsized*. The former is an identity mapping from (string) species to a numeric encode. The latter stores all the downsized (64 by 64 numpy ndarray) square images
