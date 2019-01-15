'''
Created on Aug 6, 2018

@author: abhinav.jhanwar
'''
#CLASSIFCATION/REGRESSOR PROBLEM

#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=0)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, verbose=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model.score(X_test, y_test)

# get feature importance
feat_imp = pd.Series(model.feature_importances_, feature_cols).sort_values(ascending=False)

feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()