#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error

# function used only to read a dataset
def input_dataset(file):
    return np.loadtxt(file, dtype='float')

def linearRegressionFunction(file):
    dataset = input_dataset(file)
    X = dataset[:,[0,1,3]]
    y = dataset[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train) # training model
    y_pred = lr.predict(X_test) # predicting label samples

    print("test: ", y_test, " predicted: ", y_pred)

    print(mean_absolute_error(y_test, y_pred))
    #print(cross_val_score(lr, X, y, cv=2)) # k-fold cross validation
    
linearRegressionFunction('/home/joelson/Dropbox/iris.txt')    

 
