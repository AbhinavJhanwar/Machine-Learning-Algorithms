'''
Created on Apr 18, 2017

@author: abhinav.jhanwar
'''

#Import Library
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv

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

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

validation_size = 0.1

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=validation_size, random_state=42)

# n_init: number of time centroid will be selected to make the clusters in each of the iterations to avoid local optima
# this parameter is used for the initial random initialization of centroids
model = KMeans(n_clusters=3, max_iter=1000, n_init=20)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)