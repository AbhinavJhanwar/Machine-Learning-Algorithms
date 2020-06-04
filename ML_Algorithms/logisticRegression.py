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
from sklearn import metrics, preprocessing
import csv
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

url = "../data/iris_data.csv"
    
data = pd.read_csv(url)
feature_cols = data.columns.values.tolist()[0:-1] 
X = data[feature_cols]
y = data[data.columns.values.tolist()[-1]] #names replaced by target taken from user selection
#print(X.shape)
#print(y.shape)

#preprocessing output in integers
le = preprocessing.LabelEncoder()
le.fit(y)
Encoded_classes = list(le.classes_)
y = list(map(int, le.transform(y)))

validation_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=4, stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



