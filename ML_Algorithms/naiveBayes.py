'''
Created on Apr 18, 2017

@author: abhinav.jhanwar
'''

#CLASSIFICATION

#Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
import csv
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

url = "iris_data.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection

#preprocessing output in integers
le = preprocessing.LabelEncoder()
le.fit(y)
Encoded_classes = list(le.classes_)
y = list(map(int, le.transform(y)))

validation_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=6)
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Naive bayes classification object 
model = GaussianNB() 

# Train the model using the training sets and check score
model.fit(X_train, y_train)
#Predict Output
y_pred= model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


