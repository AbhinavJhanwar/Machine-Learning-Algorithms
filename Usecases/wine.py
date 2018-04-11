'''
Created on Apr 19, 2017

@author: abhinav.jhanwar
'''

#  import pandas library
import pandas
# import scatter matrix to plot various scatter plots
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
# import various sklearn libraries
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# import various models to test
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# define dataset url
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# define feature names
names = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
# read the dataset and convert into pandas dataframe
dataset = pandas.read_csv(url,names=names)

# check shape of dataset
print("dataset shape:", dataset.shape)

# check sample of dataset
print("\n\nsample dataset:\n", dataset.head(10))

# check details of dataset
print("\n\ndataset description:\n", dataset.describe())

# check various classes counts
print("\n\ntarget variables counted:\n", dataset.groupby('class').size())

# plot scatter plot for all features
# features can be specifically mentioned to be plotted by passing the required columns
scatter_matrix(dataset)
plt.show()

# extract only values from dataset
array = dataset.values

# define X
X = array[:,1:15]

# define Y
Y = array[:,0]
validation_size = 0.20
seed = 7

# split the data in training and validation sets
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# define scoring criteria
scoring = 'accuracy'

# define various models to be tested
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
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Compare Algorithms from their cross validation scores
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# take best model and predict data
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))