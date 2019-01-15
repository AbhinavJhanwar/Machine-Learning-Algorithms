'''
Created on Aug 3, 2018

@author: abhinav.jhanwar
'''


#CLASSIFCATION/REGRESSOR PROBLEM

#Import Library
#Import other necessary libraries like pandas, numpy...
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
import csv
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#url = "C:/Users/abhinav.jhanwar/Desktop/Datasets/housing.csv"
url = "iris_data.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
    
data = pd.read_csv(url)
feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection
y = LabelEncoder().fit_transform(y)
#print(X.shape)
#print(y.shape)

validation_size = 0.25

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=0, stratify=y)

# booster = gbtree: tree-based models or gblinear: linear models
# nthread = default is max to utilize all cores available

# objective = mostly used are -
# binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
# multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
# you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
# multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.

# eval_metric = 
# rmse – root mean square error
# mae – mean absolute error
# logloss – negative log-likelihood
# error – Binary classification error rate (0.5 threshold)
# merror – Multiclass classification error rate
# mlogloss – Multiclass logloss
# auc: Area under the curve

model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=10,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27, 
 num_class= 3)

xgb_param = model.get_xgb_params()

xgtrain = xgb.DMatrix(X, y)

cvresult = xgb.cv(xgb_param, 
                  xgtrain, 
                  num_boost_round=1000,#model.get_params()['n_estimators'], 
                  nfold=5,
                  metrics='merror', 
                  early_stopping_rounds=50,
                  stratified=True)

print('\ntraining error')
print(cvresult['train-merror-mean'])
print('\nvalidation error')
print(cvresult['test-merror-mean'])

cvresult[['train-merror-mean', 'test-merror-mean']].plot()

model.set_params(n_estimators=cvresult.shape[0])

#Fit the algorithm on the data
model.fit(X_train, y_train, eval_metric='merror')
        
#Predict training set:
predictions = model.predict(X_test)
predprob = model.predict_proba(X_test)[:,1]

# Print model report:
print ("\nModel Report")
print ("Training Accuracy : %.4g" % metrics.accuracy_score(y_train, model.predict(X_train)))
print ("Testing Accuracy : %.4g" % metrics.accuracy_score(y_test, model.predict(X_test)))

                
feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')



''' PARAMETER TUNING '''

''' Tune max_depth and min_child_weight '''
# phase1 with large subset
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

gsearch1 = GridSearchCV(estimator = model,
                        param_grid = param_test1, 
                        scoring='accuracy',
                        n_jobs=2,
                        iid=False, 
                        cv=5,
                        verbose=2)
gsearch1.fit(X, y)

gsearch1.grid_scores_
gsearch1.best_params_
gsearch1.best_score_

#  Tune max_depth and min_child_weight phase 2 with smaller subset
# take parameters one above and one below the best params
param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[1,2]
}

gsearch1 = GridSearchCV(estimator = model,
                        param_grid = param_test2, 
                        scoring='accuracy',
                        n_jobs=2,
                        iid=False, 
                        cv=5,
                        verbose=2)
gsearch1.fit(X, y)

gsearch1.grid_scores_
gsearch1.best_params_
gsearch1.best_score_

# update model with best parameters
model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27, 
 num_class= 3)

''' Tune gamma '''
param_test3 = {
 'gamma':[i/10 for i in range(0,5)]
}

gsearch1 = GridSearchCV(estimator = model,
                        param_grid = param_test3, 
                        scoring='accuracy',
                        n_jobs=2,
                        iid=False, 
                        cv=5,
                        verbose=2)
gsearch1.fit(X, y)

gsearch1.grid_scores_
gsearch1.best_params_
gsearch1.best_score_

model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.4,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27, 
 num_class= 3)

''' Tune subsample and colsample_bytree '''
# phase 1
param_test4 = {
 'subsample': [i/10.0 for i in range(6,10)],
 'colsample_bytree': [i/10.0 for i in range(6,10)]
}

gsearch1 = GridSearchCV(estimator = model,
                        param_grid = param_test4, 
                        scoring='accuracy',
                        n_jobs=2,
                        iid=False, 
                        cv=5,
                        verbose=2)
gsearch1.fit(X, y)

gsearch1.grid_scores_
gsearch1.best_params_
gsearch1.best_score_

# phase 2
param_test5 = {
 'subsample': [i/100.0 for i in range(55,70,5)],
 'colsample_bytree': [i/100.0 for i in range(55,70,5)]
}

gsearch1 = GridSearchCV(estimator = model,
                        param_grid = param_test5, 
                        scoring='accuracy',
                        n_jobs=2,
                        iid=False, 
                        cv=5,
                        verbose=2)
gsearch1.fit(X, y)

gsearch1.grid_scores_
gsearch1.best_params_
gsearch1.best_score_

model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.4,
 subsample=0.55,
 colsample_bytree=0.55,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27, 
 num_class= 3)


''' Tuning Regularization Parameters '''
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

gsearch1 = GridSearchCV(estimator = model,
                        param_grid = param_test6, 
                        scoring='accuracy',
                        n_jobs=2,
                        iid=False, 
                        cv=5,
                        verbose=2)
gsearch1.fit(X, y)

gsearch1.grid_scores_
gsearch1.best_params_
gsearch1.best_score_

model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.4,
 reg_alpha=1e-5,
 subsample=0.55,
 colsample_bytree=0.55,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27, 
 num_class= 3)