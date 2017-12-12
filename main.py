"""
This function will be used to call both algorithms; Linear Regression and K-Means Clustering.
The Data set we a re working with is winequality_white.csv
The metrics we are using to evaluate our algorithms are:
Linear Regression:
K-Means Clustering:

The frame work we are using is sci-kit (sk) learn

"""
import LinearRegression
import K_meansClustering
import numpy as np
import pandas as p


def k_means_clustering(X, y):
    k_m_c = K_meansClustering.K_meansClustering(X, y)
    k_m_c.k_means_clustering_wine()


def linear_regression(X, y):
    l_r = LinearRegression.LinearRegression(X, y)
    l_r.linear_regression_wine()


def main():
    print("in main function")

    data = p.read_csv('winequality_white.csv', delimiter=';')
    X = data[data.columns[:-1]].values
    y = data[data.columns[[-1]]].values

    k_means_clustering(X, y)
    linear_regression(X, y)

    # TODO -  Step3.5: Calculate whatever metrics we need, accuracy_score, RMSE etc etc, when theyre calculated, store
    # them in 'ResultsFromAlgorithms/Results.csv'
    # TODO -  Step4: make graphs of data; Do that on python or Excel? TBC
    # TODO - Make sure your happy with the quality/readability etc of the code and that the results are good.
    # TODO - The Report - See google docs folder Breandan created


main()
