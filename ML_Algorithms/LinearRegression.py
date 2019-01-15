'''
Created on Apr 17, 2017

@author: abhinav.jhanwar
'''

#REGRESSION

#Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics, preprocessing
import statsmodels.formula.api as smf
import csv
from collections import defaultdict


url = "C:/Users/abhinav.jhanwar/Desktop/Datasets/Advertising.csv"
data = pd.read_csv(url)
#print(data.head())
#print(data.tail())
#print(data.shape)

# this produces pairs of scatterplot as shown
# use aspect= to control the size of the graphs
# use kind='reg' to plot linear regression on the graph
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
plt.show()

# create a fitted model
#lm1 = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
# print the coefficients
#print(lm1.params)
# summary
#print(lm1.summary())

feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = LinearRegression()
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

# creating csv file
predic = pd.DataFrame(data = {'Predicted_Values':y_pred}, index=X_test.index)
data = pd.concat([X_test, y_test, predic], axis=1)
print(data.columns.values.tolist())
csv_write = data
csv_write.to_csv("C:/Users/abhinav.jhanwar/Desktop/AdvertisingPrediction.csv")
