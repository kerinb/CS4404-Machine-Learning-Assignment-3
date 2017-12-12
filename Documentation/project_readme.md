#CS4404 - Machine Learning with Applications in Data Analytics#
## Assignment 3##

### Dependencies ###
sklearn/ sci-kit learn
Python Version: 3.5.2
Numpy
Pandas

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


