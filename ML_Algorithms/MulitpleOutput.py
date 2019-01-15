# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:53:53 2018

@author: abhinav.jhanwar
"""


#REGRESSION

#Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics, preprocessing
from sklearn.multioutput import MultiOutputRegressor
import statsmodels.formula.api as smf
import csv
from collections import defaultdict


url = "housing.csv"
data = pd.read_csv(url)
feature_cols = data.columns.values.tolist()[:-2]
X = data[feature_cols]
y = data[data.columns.values.tolist()[-2:]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)

# y = \beta_0 + \beta_1 \times TV + \beta_2 \times Radio + \beta_3 \times Newspaper
# print coefficient and intercept
# beta0
#print(model.intercept_)      
# beta1, beta2, beta3
#print(model.coef_)     

# pair the feature names with the coefficients
zip(feature_cols, model.coef_)
#print(list(zip(feature_cols, model.coef_)))
y_pred = model.predict(X_test)
# rmse
print("RMSE (ERROR IN PREDICTION: Preferred value: <10): ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)).mean())
print(score)

