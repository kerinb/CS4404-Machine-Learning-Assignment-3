import math
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, y):
        print('in LR init bruv...')
        self.X = X
        self.y = y
        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(X, y, test_size=0.30, shuffle=False)
        # Training: Testing = 70:30 split
        linear_regression = linear_model.LinearRegression(fit_intercept=False)
        self.lin_reg = linear_regression

    def linear_regression_wine(self):
        regression = self.lin_reg.fit(self.XTrain, self.yTrain)
        y_predict_70_30 = regression.predict(self.XTest)
        y_predict_cross_val_pred = cross_val_predict(self.lin_reg, self.X, self.y, cv=10)

        print('Coefficients: \n', regression.coef_)
        print('Variance score with 70:30: %.2f' % r2_score(self.yTest, y_predict_70_30))
        print('Variance score with CV 10: %.2f' % r2_score(self.y, y_predict_cross_val_pred))
        RMSE = math.sqrt(mean_squared_error(self.yTest, y_predict_70_30))
        print('RMSE 70/30: {}'.format(RMSE))
        RMSE = math.sqrt(mean_squared_error(self.y, y_predict_cross_val_pred))
        print('RMSE CV 10: {}'.format(RMSE))

        self.plot_my_data(y_predict_70_30, self.yTest, "70:30 VS yTest", 'red')
        self.plot_my_data(y_predict_cross_val_pred, self.y, "CV 10fold VS y", 'blue')

    def plot_my_data(self, y_predict, y_vals_on_x, num, colour):
        fig, bx = plt.subplots()

        bx.scatter(y_vals_on_x, y_predict, edgecolors=(0, 0, 0), color=colour)
        bx.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=4)
        bx.set_xlabel('Measured')
        bx.set_ylabel('Predicted')
        plt.savefig('LR plot {}.png'.format(num))
