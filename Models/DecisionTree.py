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

#url = "C:/Users/abhinav.jhanwar/Desktop/Datasets/housing.csv"
url = "C:/Users/abhinav.jhanwar/Desktop/Datasets/iris_data.csv"

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=0)

# 1. Instantiate
# default criterion=gini
# you can swap to criterion=entropy
# Create tree object 
model = DecisionTreeClassifier(criterion='gini', min_samples_split=4, random_state=0)
#model = DecisionTreeRegressor(criterion='mse', min_samples_split=18, random_state=0)    

# Define the parameter values that should be searched
sample_split_range = list(range(2, 50))
# Create a parameter grid: map the parameter names to the values that should be searched
# Simply a python dictionary
# Key: parameter name
# Value: list of values that should be searched for that parameter
# Single key-value pair for param_grid
param_grid = dict(min_samples_split=sample_split_range)

# instantiate the grid
# scoring = ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')    #classifier
best_scores = []
max=0
for k in range(0,20):
    rand = RandomizedSearchCV(model, param_grid, cv=10, scoring='accuracy', n_iter=10, random_state=5)
    rand.fit(X,y)
    best_scores.append(np.sqrt(rand.best_score_))
    if max<=best_scores[k]:
        max = best_scores[k] 
        min_samples_split = rand.best_params_['min_samples_split']
        print(min_samples_split)
    print(rand.best_params_)

model = DecisionTreeClassifier(criterion='gini', min_samples_split=min_samples_split, random_state=0)
print(np.sqrt(cross_val_score(model, X, y, scoring='accuracy', cv=10)).mean())

#grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')         #regressor

# fit the grid with data
grid.fit(X, y)

# examine the best model

# Single best score achieved across all params (min_samples_split)
print(grid.best_score_)

# Dictionary containing the parameters (min_samples_split) used to generate that score
print(grid.best_params_)    #take this min_samples_split and use in tree decision model

# Actual model object fit with those best parameters
# Shows default parameters that we did not specify
print(grid.best_estimator_)

#Predict Output
y_pred_class = grid.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred_class)#np.sqrt(metrics.mean_squared_error(y_test, y_pred_class))#
print(accuracy)