# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:18:48 2019

@author: abhinav.jhanwar
"""

# TPOT - Tree based Pipeline Optimization Technique
# pip install deap update_checker tqdm tpot

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import preprocessing 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
import sklearn.metrics

######################################## REGRESSOR
train = pd.read_csv('Big Mart Sales Train.csv')
test = pd.read_csv('Big Mart Sales Test.csv')

####################### preprocessing 

### mean imputations 
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)
test['Item_Weight'].fillna((test['Item_Weight'].mean()), inplace=True)

### reducing fat content to only two categories 
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['low fat','LF'], ['Low Fat','Low Fat']) 
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['reg'], ['Regular']) 
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['low fat','LF'], ['Low Fat','Low Fat']) 
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['reg'], ['Regular']) 


train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year'] 
test['Outlet_Establishment_Year'] = 2013 - test['Outlet_Establishment_Year'] 

train['Outlet_Size'].fillna('Small',inplace=True)
test['Outlet_Size'].fillna('Small',inplace=True)

train['Item_Visibility'] = np.sqrt(train['Item_Visibility'])
test['Item_Visibility'] = np.sqrt(test['Item_Visibility'])

col = ['Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Fat_Content']
test['Item_Outlet_Sales'] = 0

combi = train.append(test)
for item in col:
    le = LabelEncoder()
    combi[item] = le.fit_transform(combi[item])
train = combi[:train.shape[0]]
test = combi[train.shape[0]:]

test.drop('Item_Outlet_Sales',axis=1,inplace=True)

## removing id variables 
tpot_train = train.drop(['Outlet_Identifier','Item_Type','Item_Identifier'],axis=1)
tpot_test = test.drop(['Outlet_Identifier','Item_Type','Item_Identifier'],axis=1)

# separating target and features
target = tpot_train['Item_Outlet_Sales']
tpot_train.drop('Item_Outlet_Sales',axis=1,inplace=True)

# finally building model using tpot library
X_train, X_test, y_train, y_test = train_test_split(tpot_train, target,
 train_size=0.75, test_size=0.25)

# details: http://epistasislab.github.io/tpot/api/

# algos tested by tpot classifier
# 'sklearn.naive_bayes.BernoulliNB': { 'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.], 'fit_prior': [True, False] }, 
# 'sklearn.naive_bayes.MultinomialNB': { 'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.], 'fit_prior': [True, False] }, 
# 'sklearn.tree.DecisionTreeClassifier': { 'criterion': ["gini", "entropy"], 'max_depth': range(1, 11), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21) }, 
# 'sklearn.ensemble.ExtraTreesClassifier': { 'n_estimators': [100], 'criterion': ["gini", "entropy"], 'max_features': np.arange(0.05, 1.01, 0.05), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21), 'bootstrap': [True, False] },
# 'sklearn.ensemble.RandomForestClassifier': { 'n_estimators': [100], 'criterion': ["gini", "entropy"], 'max_features': np.arange(0.05, 1.01, 0.05), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21), 'bootstrap': [True, False] }, 
# 'sklearn.ensemble.GradientBoostingClassifier': { 'n_estimators': [100], 'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.], 'max_depth': range(1, 11), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21), 'subsample': np.arange(0.05, 1.01, 0.05), 'max_features': np.arange(0.05, 1.01, 0.05) },
# 'sklearn.neighbors.KNeighborsClassifier': { 'n_neighbors': range(1, 101), 'weights': ["uniform", "distance"], 'p': [1, 2] }, 
# 'sklearn.svm.LinearSVC': { 'penalty': ["l1", "l2"], 'loss': ["hinge", "squared_hinge"], 'dual': [True, False], 'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.] }, 
# 'sklearn.linear_model.LogisticRegression': { 'penalty': ["l1", "l2"], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 'dual': [True, False] }, 
# 'xgboost.XGBClassifier': { 'n_estimators': [100], 'max_depth': range(1, 11), 'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.], 'subsample': np.arange(0.05, 1.01, 0.05), 'min_child_weight': range(1, 21), 'nthread': [1] }

# preprocessing techniques
# 'sklearn.preprocessing.Binarizer': { 'threshold': np.arange(0.0, 1.01, 0.05) }, 
# 'sklearn.decomposition.FastICA': { 'tol': np.arange(0.0, 1.01, 0.05) }, 
# 'sklearn.cluster.FeatureAgglomeration': { 'linkage': ['ward', 'complete', 'average'], 'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'] }, 
# 'sklearn.preprocessing.MaxAbsScaler': { }, 
# 'sklearn.preprocessing.MinMaxScaler': { }, 
# 'sklearn.preprocessing.Normalizer': { 'norm': ['l1', 'l2', 'max'] }, 
# 'sklearn.kernel_approximation.Nystroem': { 'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'], 'gamma': np.arange(0.0, 1.01, 0.05), 'n_components': range(1, 11) }, 
# 'sklearn.decomposition.PCA': { 'svd_solver': ['randomized'], 'iterated_power': range(1, 11) }, 'sklearn.preprocessing.PolynomialFeatures': { 'degree': [2], 'include_bias': [False], 'interaction_only': [False] }, 
# 'sklearn.kernel_approximation.RBFSampler': { 'gamma': np.arange(0.0, 1.01, 0.05) }, 'sklearn.preprocessing.RobustScaler': { }, 
# 'sklearn.preprocessing.StandardScaler': { }, 'tpot.builtins.ZeroCount': { }, 
# 'tpot.builtins.OneHotEncoder': { 'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25], 'sparse': [False] } (emphasis mine)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=3, n_jobs=-1)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')

tpot_pred = tpot.predict(tpot_test)
sub1 = pd.DataFrame(data=tpot_pred)
#sub1.index = np.arange(0, len(test)+1)
sub1 = sub1.rename(columns = {'0':'Item_Outlet_Sales'})
sub1['Item_Identifier'] = test['Item_Identifier']
sub1['Outlet_Identifier'] = test['Outlet_Identifier']
sub1.columns = ['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier']
sub1 = sub1[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
sub1.to_csv('tpot.csv',index=False)

#################################### CLASSIFIER

# create train and test sets
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75, test_size=0.25, random_state=34)
tpot = TPOTClassifier(verbosity=3, 
                      scoring="accuracy", 
                      random_state=32, 
                      periodic_checkpoint_folder="tpot_results.txt", 
                      n_jobs=-1, 
                      generations=20, 
                      population_size=10,
                      early_stop=5)

tpot.fit(X_train, y_train)
tpot.fitted_pipeline_
score = tpot.score(X_test, y_test)
tpot.export('tpot_mnist_pipeline.py')  

