'''
Created on Apr 18, 2017

@author: abhinav.jhanwar
'''

#CLASSIFCATION/REGRESSION

#Import Library
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics, preprocessing
import csv
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
import matplotlib.pyplot as plt 


url = "Advertising.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection

#print(X.shape)
#print(y.shape)

#preprocessing output in integers
le = preprocessing.LabelEncoder()
le.fit(y)
Encoded_classes = list(le.classes_)
y = list(map(int, le.transform(y)))

validation_size = 0.40

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=4)
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

k_range = list(range(1, 31))

param_grid = dict()#dict(n_neighbors=k_range)

'''scores = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()'''

model = KNeighborsRegressor(n_neighbors=13)

#grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
#grid.fit(X, y)
#print(grid.cv_results_)
#print(grid.best_score_)
#print(grid.best_params_)
#print(grid.best_estimator_)

param_dist = dict(n_neighbors=range(1,31))
# n_iter controls the number of searches

# instantiate model
# 2 new params
# n_iter --> controls number of random combinations it will try
# random_state for reproducibility 
best_scores = []
max=1000
for k in range(0,20):
    rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='neg_mean_squared_error', n_iter=10, random_state=5)
    rand.fit(X,y)
    best_scores.append(np.sqrt(-rand.best_score_))
    if max>=best_scores[k]:
        max = best_scores[k] 
        print(max)
    print(rand.best_params_)
