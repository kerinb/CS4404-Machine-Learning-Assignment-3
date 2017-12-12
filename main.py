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


def main():
    print("in main function")

    data = p.read_csv('winequality_white.csv', delimiter=';')
    X = data[:-1].values
    y = data[data.columns[[-1]]].values

    K_meansClustering

    # TODO - Step3: call the functions written and stored in 'LinearRegression/LinearRegression.py' and
    # 'K_meansClustering/K_meansClustering.py'
    # TODO -  Step3.5: Calculate whatever metrics we need, accuracy_score, RMSE etc etc, when theyre calculated, store
    # them in 'ResultsFromAlgorithms/Results.csv'
    # TODO -  Step4: make graphs of data; Do that on python or Excel? TBC
    # TODO - Make sure your happy with the quality/readability etc of the code and that the results are good.
    # TODO - The Report - See google docs folder Breandan created


main()
