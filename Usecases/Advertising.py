'''
Created on Jul 27, 2017

@author: abhinav.jhanwar
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics, preprocessing
import statsmodels.formula.api as smf
import csv
from collections import defaultdict


url = "C:/Users/abhinav.jhanwar/Desktop/Datasets/Advertising.csv"
data = pd.read_csv(url, index_col = 0)
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

# creating csv file
predic = pd.DataFrame(data = {'Predicted_Values':y_pred}, index=y_test.index)
data = pd.concat([X_test, y_test, predic], axis=1)
csv_write = pd.DataFrame(data = data)
csv_write.to_csv("C:/Users/abhinav.jhanwar/Desktop/AdvertisingPrediction.csv")


'''

############# Handling Categorical Features with Two Categories###########
# set a seed for reproducibility
np.random.seed(12345)

# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(data))
mask_large = nums>0.5

# initially set Size to small, then change roughly half to be large
data['size'] = 'large'

# Series.loc is a purely label-location based indexer for selection by label
data.loc[mask_large, 'size'] = 'small'

print(data['size'])

le = preprocessing.LabelEncoder()
le.fit(data['size'])
Encoded_classes = list(le.classes_)
data['size'] = list(map(int, le.transform(data['size'])))
print(data['size'])

feature_cols = ['TV', 'Radio', 'Newspaper', 'size']
X = data[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

zip(feature_cols, model.coef_)
print(list(zip(feature_cols, model.coef_)))
'''


