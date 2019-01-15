'''
Created on Aug 1, 2018

@author: abhinav.jhanwar
'''

'''
R-Square: It determines how much of the total variation in Y (dependent variable) 
is explained by the variation in X (independent variable). 
Mathematically, it can be written as:
    
R-Square = 1-(sum(y-y_pred)^2/sum(y-y_mean)^2)

The value of R-square is always between 0 and 1,
where 0 means that the model does not model explain any variability in the target 
variable (Y) and 1 meaning it explains full variability in the target variable.
'''


''' RIDGE REGRESSION '''
# uses L2 regularization technique

# used for regularization
# alpha determines the lambda - penalty value
# used when model is overfitting
# helps in reducing the weights assigned to various features
# as penalty increases cost function increases and as a result while calculating gradient descent, theta value - weights, reduced to reduce the cost

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

url = "Advertising.csv"
data = pd.read_csv(url)

feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# higher the value of alpha, higher the value of penalty lambda
model = Ridge(alpha=0.5, normalize=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = np.mean((y_pred - y_test)**2)

model.score(X_test, y_test)


''' LASSO REGRESSION (Least Absolute Shrinkage Selector Operator) '''
# uses L1 regularization technique

''' In Ridge algorithm We can see that as we increased the value of alpha, 
coefficients were approaching towards zero, 
but if you see in case of lasso, even at smaller alpha’s, 
our coefficients are reducing to absolute zeroes. 
Therefore, lasso selects the only some feature while reduces the coefficients of others to zero. 
This property is known as feature selection and which is absent in case of ridge.'''

'''Mathematics behind lasso regression is quiet similar to that of ridge 
only difference being instead of adding squares of theta, 
we will add absolute value of Θ in cost function.'''



url = "Advertising.csv"
data = pd.read_csv(url)

feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# higher the value of alpha, higher the value of penalty lambda
model = Lasso(alpha=0.01, normalize=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = np.mean((y_pred - y_test)**2)

model.score(X_test, y_test)




'''Elastic Net Regression '''

# generally works well with large dataset around 10000 records & large features
# hybrid of redge & lasso regression models
# combination of both L1 and L2 regularization and 
# hence adds absolute value of theta + squared value of theta in cost function





url = "Advertising.csv"
data = pd.read_csv(url)

feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# higher the value of alpha, higher the value of penalty lambda
# a,b = lambda for L1, L2 respectively
# alpha = a+b and l1_ratio = a/(a+b)
# hence we set alpha & l1_ratio to control lasso & redge

model = ElasticNet(alpha=0.01, l1_ratio=0.5, normalize=False)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = np.mean((y_pred - y_test)**2)

model.score(X_test, y_test)







