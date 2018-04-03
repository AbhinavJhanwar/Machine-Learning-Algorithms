'''
Created on Apr 28, 2017

@author: abhinav.jhanwar
'''

import pandas
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

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
names = ['buying','maint','doors','persons','lug_boot','safety','class']
dataset = pandas.read_csv(url, names=names)

#print(dataset.shape)
#print(dataset.head(10))            
#print(dataset.describe())            #gives statistical details
#print(dataset.groupby('class').size())

array = dataset.values

X = array[:,0:6]
Y = array[:,6]
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
    
model = SVC()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

cs = open("C:/Users/abhinav.jhanwar/Desktop/TUTORIALS/adults.csv","w+")
cs.write("Y_validations,predictions\n")
for i in range(0,len(predictions)):
    cs.write(Y_validation[i])
    cs.write(",")
    cs.write(predictions[i])
    cs.write("\n")
cs.close()
#print("validations: ",Y_validation)
#print("predictions: ",predictions)'''
