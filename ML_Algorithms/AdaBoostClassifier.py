'''
Created on Apr 17, 2017

@author: abhinav.jhanwar
'''

#CLASSIFICATION/REGRESSION PROBLEM

#Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
import csv
from sklearn.linear_model import LinearRegression

url = "housing.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection
#print(X.shape)
#print(y.shape)

'''#preprocessing output in integers
le = preprocessing.LabelEncoder()
le.fit(y)
Encoded_classes = list(le.classes_)
y = list(map(int, le.transform(y)))'''

validation_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=10)

# Instantiate
abc = AdaBoostRegressor()

# Fit
abc.fit(X_train, y_train)

# Predict
y_pred = abc.predict(X_test)

accuracy = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print(accuracy)
