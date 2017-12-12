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
Algorithm2: Support Vector Regression

### Metrics ###
Metric1: R2score
Metric2: ExplainedVar
Metric3: MeanAbsoluteError
Metric4: MeanSquaredError
Metric5: RootMeanSquaredError
Metric6: MeanSquaredLogError
Metric7: MedianAbsoluteError
Metric8: Accuracy

### Code Explained ###
In the main.py file, the data is read in from the data set; winequality_white.csv, using the <i>load_data_from_csv_file<.i>
function in the SharedFunctions python file, the line <i> data = [item for item in load_data_from_csv_file(4898, 'winequality_white.csv')]</i>
reads in the entire data set from the data set given.
This is then converted into the data that we will use on our algorithms, the X and y values.

For this project, the team are not going to use any data pre-processing; for example feature selection, as each of the
data features are considered important for implementing an effective predictive model. The team had implemented a function
to check the Pearsons coefficient for each of the features compared to the dependent variable, this determines a linear
coefficient, determining whether the independant variables have a strong/weak correlation with the independent variable. From this function, it was determined
that all of the features were deemed important for the purposes of this investigation at with all features used, the highest accuracy
was calculated.

### Linear Regression ###
Two arrays are passed into this class, x and yl ie the dependent and independent variables. These are stored as attributes
of the class <i> self.x = x</i> etc. The test and training values are determined by splitting the x and y arrays into
seperate arrays, this was done using <i>self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(x, y, test_size=0.30, shuffle=False)</i>
which created xTrain, xTest, yTrain and yTest arrays to train and test the linea regression.

Each algorithm we are using is stored in a seperate python file and is represented as a class; LinearRegression class etc
The constructor of each class is called from the main file and the X and y data are passed to the constructor. In the
constructor, the X and y data are then split into a 70:30 ratio for test and training data using
<i>self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.30, shuffle=False)</i>, which
is a method provided to us by the <i>sklearn.model_selection</i> library. The reason each of these arrays of data were stored as
attributes of the class is to simplify passing data through functions.
The Linear Regression was defined using the following line of code:
<i>self.lin_reg = linear_model.LinearRegression(fit_intercept=False)</i>

the linear regression was trained using:
<i>regression = self.lin_reg.fit(self.x_train, self.y_train)</i>

and made its predictions as follows:
<i>predict = cross_val_predict(regression, self.x_test, self.y_test, cv=10)

Several metrics for determining the the correctness of the algorithm were highlighted above and were determined in the
<i>cacl_results</i> function in SharedFunctions file.


### Support Vector Regression ###
Much like above in Linear Regression, the same method of 70:30 data split etc was used for creating the training and testing values.
The following lines define how the support vector regression algorithm is implemented, trained and makes its predictions respectively:

<i>clf = SVR(kernel="rbf", C=10, epsilon=0.25)</i>
<i>self.svr_ = clf.fit(scaler.transform(self.XTrain), self.yTrain)</i>
<i>self.pred = clf.predict(scaler.transform(self.XTest))</i>

### Comment on results ###
the results obtained were ok, the accuracy for the linear regression was 0.49....,  which could be better, but at this moment,
I am not too sure how to improve this value, as we have tried to implement feature selection, which led the team to believe that
using all features in the data would result in the most accurate results. Ifwe have time, we should look into this more!

for the SVR, the results were a bit better, yielding 0.56..., which is approx. 7% better! Again, this could be better, and if time permits,
the team will look into improving on this result with possibly more pre-processing of the data.

A note on the Root Mean square error, for the linear regression, the RMSE was too high for comfort really at 0.825. For the
SVR, the RMSE value was more acceptable at 0.517, but still not great. THe team should look into improving both of these values
with pre-processing the data, possibly by applying data normalisation prior to training the algorithms.

In an ideal world, another possibility that could allow us to obtain a better result would be to get a larger data set, which,
at the moment is un-attainable.... Maybe we could become Sommeliers and make our own data set??

