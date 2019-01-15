'''
Created on Apr 17, 2017

@author: abhinav.jhanwar
'''

#CLASSIFCATION/REGRESSOR PROBLEM

#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
import csv
from sklearn import metrics
import numpy as np

#url = "housing.csv"
url = "iris_data.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection
#print(X.shape)
#print(y.shape)

validation_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=0, stratify=y)

# 1. Instantiate
# default criterion=gini
# you can swap to criterion=entropy
# Create tree object 
model = DecisionTreeClassifier(criterion='gini', min_samples_split=4, random_state=0)
#model = DecisionTreeRegressor(criterion='mse', min_samples_split=18, random_state=0)    

model.fit(X_train, y_train)

#Predict Output
y_pred_class = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred_class)#np.sqrt(metrics.mean_squared_error(y_test, y_pred_class))#
print(accuracy)