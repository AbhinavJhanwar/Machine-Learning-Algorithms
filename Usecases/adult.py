'''
Created on Apr 28, 2017

@author: abhinav.jhanwar
'''

import pandas
import numpy
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
import csv
from sklearn import preprocessing
from collections import defaultdict
from _codecs import encode

def find_type(value):
    try:
        var_type = type(int(value))
        var_type = type(0.0)
    except ValueError:
        try:
            var_type = type(float(value))
        except ValueError:
            var_type = type(value)
    return str(var_type)

url = "C:/Users/abhinav.jhanwar/Desktop/adult_tut.csv"
        
with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)
        
columns = defaultdict(list)
encoded_columns = defaultdict(list)
    
with open(url) as csvFile:
    reader = csv.DictReader(csvFile)
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
            
le = preprocessing.LabelEncoder()
for item in names:
    item_type = find_type(columns[item][1])
    if 'str' in item_type:
        le.fit(columns[item])
        Encoded_classes = list(le.classes_)
        encoded_columns[item] = list(map(int, le.transform(columns[item])))
    else:
        encoded_columns[item] = list(map(int, columns[item]))
        #print(Encoded_classes)
        #print(item, "to be converted")
        
dataset = pandas.DataFrame(encoded_columns, columns = names)
array = dataset.values


X = array[:,0:14] 
Y = array[:,14]
#print(dataset.head(10))

min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_minmax, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names1 = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names1.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

cs = open("C:/Users/abhinav.jhanwar/Desktop/adults.csv","w+")
cs.write("Y_validations,predictions\n")
for i in range(0,len(predictions)):
    cs.write(str(Y_validation[i]))
    cs.write(",")
    cs.write(str(predictions[i]))
    cs.write("\n")
cs.close()
#print("validations: ",Y_validation)
#print("predictions: ",predictions)'''
