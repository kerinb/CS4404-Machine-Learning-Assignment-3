import LinearRegression
import SupportVectorRegression
import numpy as np
from SharedFunctions import load_data_from_csv_file, determine_features_to_use


def support_vector_regression(X, y):
    l_r = SupportVectorRegression.SupportVectorRegression(X, y)
    l_r.support_vector_regression_wine()


def linear_regression(X, y):
    l_r = LinearRegression.LinearRegression(X, y)
    l_r.linear_regression_wine()


def main():
    print("in main function")
    init = str("R2score, ExplainedVar, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, MeanSquaredLogError," \
               "MedianAbsoluteError, Accuracy, Algorithm\n")
    text_file = open("Output.txt", "w")
    text_file.write(init)
    text_file.close()

    data = [item for item in load_data_from_csv_file(4898, 'winequality_white.csv')]
    features_to_include = determine_features_to_use(data)
    X = [None] * len(features_to_include)

    for j in range(len(features_to_include)):
        print(features_to_include.__getitem__(j))
        X[j] = [
            np.array([data[i][features_to_include.__getitem__(j)]])
            for i in range(len(data))]
    X = np.array(X)
    X = X[:, :, 0].T
    y = np.array([data[i][11] for i in range(len(data))])

    print('x data', X.shape, ' y data', y.shape)
    linear_regression(X, y)
    support_vector_regression(X, y)


main()
