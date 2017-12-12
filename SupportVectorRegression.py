import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from SharedFunctions import plot_my_data, calc_results


class SupportVectorRegression:
    def __init__(self, X, y):
        print('in LR init bruv...')
        self.X = X
        self.y = y
        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(X, y, test_size=0.3, random_state=6)
        scaler = StandardScaler().fit(self.XTrain)
        clf = SVR(kernel="rbf", C=10, epsilon=0.2)
        self.svr_ = clf.fit(scaler.transform(self.XTrain), self.yTrain)
        self.pred = clf.predict(scaler.transform(self.XTest))
        # y_rand = np.random.randint(11, size=len(y_test))

    def support_vector_regression_wine(self):
        pred_int = np.array([round(y) for y in self.pred], dtype=np.int32)
        y_test_int = np.array([round(y) for y in self.yTest], dtype=np.int32)

        sum_of_correct_predits = 0
        for i in range(len(pred_int)):
            if self.yTest[i] == pred_int[i]:
                # print(self.yTest[i], pred_int[i])
                sum_of_correct_predits += 1

        print("\n\nWith int output for SVR")
        calc_results(y_test_int, pred_int, 'svr')
        plot_my_data(self.y, self.pred, self.yTest, "CV 10fold VS y", 'blue', 'svr')
