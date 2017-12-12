from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import linear_model


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
