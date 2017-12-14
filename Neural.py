import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas
from SharedFunctions import calc_results, plot_my_data


def neural_network():

    # Preprocessing data
    filename = 'winequality-white.csv' 
    data_read = pandas.read_csv(filename, delimiter = ';')
    print ('Shape of data is ' , '\n' , data_read.shape)

    df=pandas.DataFrame(data_read)
    df = df.describe()
    print (df)

    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    for i in features:
	    data = data_read[str(i)]
	    plt.hist(data, bins='auto')  
	    plt.title(str(i) + "Histogram")
	    plt.xlabel('Value')
	    plt.show()
    data = data_read['quality']  
    plt.hist(data, bins='auto') 

    #Feature Selection

    for i in features:
	    data1 = data_read[str(i)]
	    data2 = data_read['quality']
	    cor = np.corrcoef(data1, data2)[0,1]  
	    print ('Quality correlation with', str(i), '::',  cor)
    
    # Adding a column of ones 
    m=data_read.shape[0]
    data_read = np.column_stack((np.ones((m, 1)), data_read))
    print ('new data shape', data_read.shape)

    # after getting all features, we add/remove features which are of less importance
    data_to_use = np.array(data_read)
    print ('Shape of data after converted to numpy array ', '\n' , data_to_use.shape)
    X = data_to_use[:,[0,1,2,4,6,9,10,11]]
    print ('Printing X', '\n',  X)
    print ('Shape of X','\n', X.shape)

    y = data_to_use[:,12]
    print ('printing y' , '\n',  y)
    print ('y.shape', '\n', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Normalizing the data for use
    # scaler object has saved means aand standard deviations
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    print (X.mean(axis=0)) #0

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Applying Neural Network to train the model using cross validation

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=30, solver = 'adam', activation='logistic', learning_rate_init=.01)


    # Cross validation score
    print ('Cross validation score:')
    scores = cross_val_score(mlp, X_train,y_train, cv=10) 
    print (scores.mean())

    print ('Cross validation f1 macro score:')
    scores = cross_val_score(mlp, X_train,y_train, cv=10, scoring='f1_macro')  
    print (scores.mean())

    # Fitting the model
    mlp = mlp.fit(X_train, y_train)
    
    #metrics
    predicted = mlp.predict(X_test)
    calc_results(y_test, predicted, 'neural_network')
    plot_my_data(y, predicted, y_test, "CV 10fold VS y", 'blue', 'neural_network')
   

    
    
   
