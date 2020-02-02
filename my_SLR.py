#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 18:39:58 2019

@author: joy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')


# create numpy matrix of independant variables
X = dataset.iloc[:,:-1].values
# alternative- dataset.iloc[:,:-1]

# NOTE: if dont put .values, it would be a pandas array
Y = dataset.iloc[:,-1].values

# take care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# encode the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# split the dataset into the Training set and Test Set
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train, Y_train)

# predict the test set results
y_pred = regressor.predict(X_test)

#Visualizing the training set fitting results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set fitting results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()