#CS4404 - Machine Learning with Applications in Data Analytics#
## Assignment 3##

### Dependencies ###
sklearn/ sci-kit learn
Python Version: 3.5.2
Numpy
Pandas
scipy
matplotlib
python3-tk

### Algorithms ###
Algorithm1: Linear Regression
Algorithm2: K-means Clustering

### Metrics ###
Metric1: ??????????????????????
Metric2: ??????????????????????

### Code Explained ###
In the main.py file, the data is read in from the data set; winequality_white.csv, using the pandas library, and more
specifically - <i> data = p.read_csv('winequality_white.csv', delimiter=';') </i>, this reads in the entire data set.
This is then converted into the data that we will use on our algorithms, the X and y sets of dependant and independent
variables by obtaining the <i> values</i> in the pandas array; <i>X = data[:-1].values</i>

For this project, the team are not going to use any data pre-processing; for example feature selection, as each of the
data features are considered important for implementing an effective predictive model.

Each algorithm we are using is stored in a seperate python file and is represented as a class; LinearRegression class etc
The constructor of each class is called from the main file and the X and y data are passed to the constructor. In the
constructor, the X and y data are then split into a 70:30 ratio for test and training data using
<i> self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(X, y, test_size=0.30, shuffle=False)</i>, which
is a method provided to us by the <i>sklearn.model_selection</i> library.