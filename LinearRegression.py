from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, X, y):
        print('in LR init bruv...')
        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(X, y, test_size=0.30, shuffle=False)
        # Training: Testing = 70:30 split

    def linear_regression_wine(self):
        pass
