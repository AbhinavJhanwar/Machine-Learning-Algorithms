'''
Created on Apr 17, 2017

@author: abhinav.jhanwar
'''

# Import Library
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import csv
from sklearn.metrics import accuracy_score
import seaborn as sns

# Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=42)

# Create SVM classification object 
# 1. Instantiate
# Default kernel='rbf' : Radial Basis Function
# We can change to others
# kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
# linear = no kernal; when features are less and samples are more, it is like logistic regression
model = SVC(kernel='linear', random_state=42)
# try for all kernels and take which one gives best result.
# Train the model using the training sets and check score
model.fit(X_train, y_train)

param_grid = {
    'C': [1e3, 5e3, 5e4, 1e5],
    'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    #'kernel' :['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
}

# GridSearch
grid = GridSearchCV(model, param_grid)
grid.fit(X_train, y_train)
print(grid.best_estimator_)

y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


