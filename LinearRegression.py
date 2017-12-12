import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import linear_model
from SharedFunctions import plot_my_data, calc_results


class LinearRegression:
    def __init__(self, x, y):
        print('\n\nin LR init bruv...')
        self.X = x
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.30, shuffle=False)
        self.lin_reg = linear_model.LinearRegression(fit_intercept=False)

    def linear_regression_wine(self):
        regression = self.lin_reg.fit(self.x_train, self.y_train)
        predict = cross_val_predict(regression, self.x_test, self.y_test, cv=10)

        predicted_int = np.array([y for y in predict], dtype=np.int32)
        y_test_int = np.array([y for y in self.y_test], dtype=np.int32)

        sum_of_correct_predicts = 0
        for i in range(len(predicted_int)):
            if self.y_test[i] == predicted_int[i]:
                sum_of_correct_predicts += 1

        print("With int output for Linear Regression")
        calc_results(y_test_int, predicted_int, 'lin_reg')
        plot_my_data(self. y, predict, self.y_test, "CV 10fold VS y", 'blue', 'lin_reg')
        print('made plot ')
