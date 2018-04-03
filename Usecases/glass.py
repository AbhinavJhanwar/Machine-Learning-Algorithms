'''
Created on Apr 28, 2017

@author: abhinav.jhanwar
'''

import pandas
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "C:/Users/abhinav.jhanwar/Desktop/Datasets/glass1.csv"
names = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type of glass']
dataset = pandas.read_csv(url, names=names)
#print(dataset.shape)
#print(dataset.head(10))            
#print(dataset.describe())            #gives statistical details
#print(dataset.groupby('class').size())

array = dataset.values

X = array[:,0:9]
Y = array[:,9]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

'''model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
    
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))'''

model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
predic = np.array([1.51711,14.23,0.00,2.08,73.36,0.00,8.62,1.67,0.00])
print(model.predict(predic))
print(accuracy_score(Y_validation, predictions))

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

model = SVC()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

cs = open("C:/Users/abhinav.jhanwar/Desktop/glass.csv","w+")
cs.write("Y_validations,predictions\n")
for i in range(0,len(predictions)):
    cs.write(str(Y_validation[i]))
    cs.write(",")
    cs.write(str(predictions[i]))
    cs.write("\n")
cs.close()
#print("validations: ",Y_validation)
#print("predictions: ",predictions)'''
