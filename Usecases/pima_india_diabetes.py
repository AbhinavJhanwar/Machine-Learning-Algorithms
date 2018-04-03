'''
Created on Apr 17, 2017

@author: abhinav.jhanwar
'''

#CLASSIFICATION Problem

#Import Library
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer

def scaleData(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    rescaledX = scaler.fit_transform(X)
    return rescaledX

def standerdizeData(X):
    scaler = StandardScaler().fit(X)
    rescaledX = scaler.transform(X)
    return rescaledX

def normalizeData(X):
    scaler = Normalizer().fit(X)
    normalizedX = scaler.transform(X)
    return normalizedX

def binarizeData(X):
    scaler = Binarizer(threshold=0.0).fit(X)
    binarizedX = scaler.transform(X)
    return binarizedX

url = "C:/Users/abhinav.jhanwar/Desktop/Datasets/pima-indians-diabetes.csv"
    
data = pd.read_csv(url)
names = data.columns.values.tolist()
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection
#print(X.shape)
#print(y.shape)

X = standerdizeData(X)
X = scaleData(X)
#X = normalizeData(X)
#X = binarizeData(X)

validation_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

