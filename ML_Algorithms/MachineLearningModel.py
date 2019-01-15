'''
Created on Jul 31, 2017

@author: abhinav.jhanwar
'''

#Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
#import algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import csv
from sklearn.preprocessing import LabelEncoder, Imputer
import warnings


#url as taken from user
#returns names to be displayed on page2
#page1
def getDataset(url):        
    data = pd.read_csv(url)
    names = data.columns.values.tolist()
    return names, data

#getdata-->getTargetFeature
#features --> user selected features
#target --> user selected target
#page2
def getTargetFeature(data, features, target):
    global featureType
    X = data[features]
    y = data[target] 
    
    problemType = discreteContinuous(y)
    
    le = LabelEncoder()
    encodeClasses = dict()
            
    featureType = X.dtypes
    for item in features:
        # Encoding only categorical variables
        if featureType[item] == 'object':
            X[item] = le.fit_transform(X[item])
            if ' ?' in le.classes_ or 'NaN' in le.classes_:
                encodeClasses[item] = True
            else:
                encodeClasses[item] = False
    
    #X = imputeMissingValues(X, featureType, encodeClasses)
        
    return X, y, problemType

#get type of target
def discreteContinuous(target):
    uniquePercent = len(np.unique(target))/len(target)
    if(uniquePercent>0.2):
        return 'Regression'
    else:
        return 'Classification'

#takes input as type of target to be predicted
#returns models according to type of target
#page3    
def getModels(problemType):
    if(problemType == 'Classification'):
        return getClassificationModels()
    else:
        return getRegressionModels()

def getClassificationModels():
    models = []
    models.append(('LogisticRegression', LogisticRegression()))
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))     #KNN
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier())) #CART
    models.append(('NaiveBayes', GaussianNB()))
    #models.append(('SupportVectorMachine', SVC()))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    return models

def getRegressionModels():
    models = []
    models.append(('LinearRegression', LinearRegression()))
    models.append(('KNeighborsRegressor', KNeighborsRegressor()))     #KNN
    models.append(('DecisionTreeRegressor', DecisionTreeRegressor())) #CART
    models.append(('AdaBoostRegressor', AdaBoostRegressor()))
    return models

#returns best model with highest accuracy score/lowest RMSE value
def getBestModel(models, problemType):
    if problemType == 'Regression':
        bestModel, RMSEScore = getBestRegressionModel(models)
        return bestModel, RMSEScore
    elif problemType == 'Classification':
        bestModel, Accuracy = getBestClassificationModel(models)
        return bestModel, Accuracy
        
def getBestRegressionModel(models):
    RMSEScore, score = 1000,0
    
    for modelName, model in models:
        if modelName == 'LinearRegression':
            score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
        elif modelName == 'KNeighborsRegressor':
            score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
        elif modelName == 'DecisionTreeRegressor':
            score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
        elif modelName == 'AdaBoostRegressor':
            score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
        #print("Model:",modelName," Score:",score)
        if score<RMSEScore:
            RMSEScore = score
            bestModel = modelName
            
    return bestModel, RMSEScore
        
def getBestClassificationModel(models):
    Accuracy, score = 0, 0
    
    for modelName, model in models:
        if modelName == 'LogisticRegression':
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
        elif modelName == 'LinearDiscriminantAnalysis':
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
        elif modelName == 'KNeighborsClassifier':
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
        elif modelName == 'DecisionTreeClassifier':
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
        elif modelName == 'NaiveBayes':
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
        elif modelName == 'SupportVectorMachine':
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
        elif modelName == 'AdaBoostClassifier':
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
        
        #print("Model:",modelName," Score:",score)
        if score>Accuracy:
            Accuracy = score
            bestModel = modelName
            
    return bestModel, Accuracy

def getScore(problemType, modelName):
    if problemType == 'Classification':
        return getAccuracy(modelName)
    elif problemType == 'Regression':
        return getRMSE(modelName)
    
def getAccuracy(modelName):
    score = 0
    
    for name, model in models:
        if modelName == name:
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((cross_val_score(model, X, y, scoring='accuracy', cv=5)).mean())
            return score
    
def getRMSE(modelName):
    score = 0
    for name, model in models:
        if modelName == name:
            score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
            return score
        elif modelName == name:
            score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
            return score            
    
#returns predicted values for uploaded csvfile or user entered data
def getPredictedValue(bestModel, test):
    model = None
    
    if bestModel == 'LinearRegression':
        model = LinearRegression()
        model.fit(X, y)
        
        
    elif bestModel == 'KNeighborsRegressor':
        param_dist = dict(n_neighbors=range(1,31))
        model = KNeighborsRegressor(n_neighbors=13)
        best_scores = 0
        min_score=1000
        for k in range(0,20):
            rand = RandomizedSearchCV(model, param_dist, cv=5, scoring='neg_mean_squared_error', n_iter=5, random_state=5)
            rand.fit(X, y)
            best_scores = (np.sqrt(-rand.best_score_))
            if min_score>=best_scores:
                min_score = best_scores 
                n_neighbors = rand.best_params_['n_neighbors']
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X, y)
        
            
    elif bestModel == 'DecisionTreeRegressor':
        param_dist = dict(min_samples_split=list(range(2, 50)))
        model = DecisionTreeRegressor(criterion='mse', min_samples_split=4, random_state=0)
        best_scores = 0
        min_score=1000
        for k in range(0,20):
            rand = RandomizedSearchCV(model, param_dist, cv=5, scoring='neg_mean_squared_error', n_iter=5, random_state=5)
            rand.fit(X,y)
            best_scores = (np.sqrt(-rand.best_score_))
            if min_score>=best_scores:
                min_score = best_scores 
                min_samples_split = rand.best_params_['min_samples_split']
        model = DecisionTreeRegressor(criterion='mse', min_samples_split=min_samples_split, random_state=0)
        model.fit(X, y)
       
        
    elif bestModel == 'AdaBoostRegressor':
        model = AdaBoostRegressor()
        model.fit(X, y)
        
    
    elif bestModel == 'LogisticRegression':
        model = LogisticRegression()
        model.fit(X, y)
        
        
    elif bestModel == 'LinearDiscriminantAnalysis':
        model = LinearDiscriminantAnalysis()
        model.fit(X, y)
        
    
    elif bestModel == 'KNeighborsClassifier':
        param_dist = dict(n_neighbors=range(1,31))
        model = KNeighborsClassifier(n_neighbors=13)
        best_scores = 0
        max_score = 0
        for k in range(0,20):
            rand = RandomizedSearchCV(model, param_dist, cv=5, scoring='accuracy', n_iter=5, random_state=5)
            rand.fit(X,y)
            best_scores = (np.sqrt(rand.best_score_))
            if max_score<=best_scores:
                max_score = best_scores 
                n_neighbors = rand.best_params_['n_neighbors']
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X, y)
        
        
    elif bestModel == 'DecisionTreeClassifier':
        param_dist = dict(min_samples_split=list(range(2, 50)))
        model = DecisionTreeClassifier(criterion='gini', min_samples_split=4, random_state=0)
        best_scores = 0
        max_score = 0
        for k in range(0,20):
            rand = RandomizedSearchCV(model, param_dist, cv=5, scoring='accuracy', n_iter=5, random_state=5)
            rand.fit(X,y)
            best_scores = (np.sqrt(rand.best_score_))
            if max_score<=best_scores:
                max_score = best_scores 
                min_samples_split = rand.best_params_['min_samples_split']
        model = DecisionTreeClassifier(criterion='gini', min_samples_split=min_samples_split, random_state=0)
        model.fit(X, y)
        
        
    elif bestModel == 'NaiveBayes':
        model = GaussianNB()
        model.fit(X, y)
       
    
    elif bestModel == 'SupportVectorMachine':
        param_dist = {'C': [1e3, 5e3, 5e4, 1e5], 'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
        model = SVC(kernel='linear', random_state=42)
        best_scores = 0
        max_score = 0
        for k in range(0,20):
            rand = RandomizedSearchCV(model, param_dist, cv=5, scoring='accuracy', n_iter=5, random_state=5)
            rand.fit(X,y)
            best_scores = (np.sqrt(rand.best_score_))
            if max_score<=best_scores:
                max_score = best_scores 
                C = rand.best_params_['C']
                gamma = rand.best_params_['gamma']
        model = SVC(kernel='linear', random_state=42, C=C, gamma=gamma)
        model.fit(X, y)
       
            
    elif bestModel == 'AdaBoostClassifier':
        model = AdaBoostClassifier()
        model.fit(X, y)
        
    y_pred = model.predict(test)
    
    return y_pred

def imputeMissingValues(X, featureType, encodeClasses):
    objectDF = pd.DataFrame()
    othersDF = pd.DataFrame()

    for item in X.columns.values.tolist():      #replace with features
        if featureType[item] == 'object':
            if encodeClasses[item]:
                objectDF[item] = X[item]
        else:
            othersDF[item] = X[item]
    
    objectDFFeatures = objectDF.columns.values.tolist()
    othersDFFeatures = othersDF.columns.values.tolist()
    
    imp = Imputer(missing_values=0, strategy='most_frequent', axis=0)
    objectDF = imp.fit_transform(objectDF)
    objectDF = pd.DataFrame(data=objectDF, columns=objectDFFeatures)
    
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    othersDF =imp.fit_transform(othersDF)
    othersDF = pd.DataFrame(data=othersDF, columns=othersDFFeatures)
    
    for item in X.columns.values.tolist():               #replace with features
        if featureType[item] == 'object':
            if encodeClasses[item]:
                X[item] = objectDF[item]
        else:
            X[item] = othersDF[item]
    return X

#for page1
warnings.filterwarnings("ignore")
names,data = getDataset("housing.csv")    #names-->to be displayed on page2

#for page2
X,y,problemType = getTargetFeature(data, names[0:-1], names[-1])

#for page3
models = getModels(problemType)
# for page4
model, Score = getBestModel(models, problemType)
print("Model:",model,"Score:", Score)
# for page5
# test is entered or uploaded by user
#test = ['230.1', '37.8', '69.2']
#test = ['1.51763', '12.8', '3.66', '1.27', '73.01', '0.6', '8.56', '0', '0']
#test = ['1.52119', '12.97', '0.33', '1.51', '73.39', '0.13', '11.27', '0', '0.28']

#testDF = pd.DataFrame(test)
#print(len(testDF))
#print()

#y_pred = makePredictions(X, y, model, test)

#print(y_pred)
# creating csv file
'''predic = pd.DataFrame(data = {'Predicted_Values':y_pred}, index=test.index)
data = pd.concat([test, y_pred], axis=1)
csv_write = pd.DataFrame(data = data)
csv_write.to_csv("AdvertisingPrediction.csv")'''