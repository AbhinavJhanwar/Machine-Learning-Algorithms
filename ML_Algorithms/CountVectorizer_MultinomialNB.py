'''
Created on Jul 31, 2017

@author: abhinav.jhanwar
'''
# 1. import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# 2. instantiate CountVectorizer (vectorizer)
vect = CountVectorizer()

# example text for model training (SMS messages)
simple_train = ['call you tonight', 'Call me a cab', 'please call me.. please']

# 3. fit
# learn the 'vocabulary' of the training data (occurs in-place)
vect.fit(simple_train)

# examine the fitted vocabulary
#print(vect.get_feature_names())

# 4. transform training data into a 'document-term matrix'
simple_train_dtm = vect.transform(simple_train)
#print(simple_train_dtm.toarray())

# examine the vocabulary and document-term matrix together
# pd.DataFrame(matrix, columns=columns)
data = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
#print(data)

# check the type of the document-term matrix
#print(type(simple_train_dtm))

# examine the sparse matrix contents
# left: coordinates of non-zero values
# right: values at that point
# CountVectorizer() will output a sparse matrix
#print('sparse matrix')
#print(simple_train_dtm)

#print('dense matrix')
#print(simple_train_dtm.toarray())

# example text for model testing
simple_test=['Please don\'t call me']

# 4. transform testing data into a document-term matrix (using existing vocabulary)
simple_test_dtm=vect.transform(simple_test)
#print(simple_test_dtm.toarray())

# examine the vocabulary and document-term matrix together
data = pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())
#print(data)

# alternative: read file into pandas from a URL
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
features = ['label', 'message']
sms = pd.read_table(url, header=None, names=features)

# examine the shape
#print(sms.shape)

# examine the first 10 rows
#print(sms.head())

# examine the class distribution
#print(sms.label.value_counts())

# convert label to a numerical variable
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})

# check that the conversion worked
#print(sms.head())

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
#print(X.shape)
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.fit_transform(X_train)

X_test_dtm = vect.transform(X_test)

model = MultinomialNB()
model.fit(X_train_dtm, y_train)

y_pred_class = model.predict(X_test_dtm)

# calculate accuracy of class predictions
accuracy = metrics.accuracy_score(y_test, y_pred_class)
#print(accuracy)

# examine class distribution
#print(y_test.value_counts())

# calculate null accuracy (for multi-class classification problems)
# .head(1) assesses the value 1208
null_accuracy = y_test.value_counts().head(1)/len(y_test)
#print('Null accuracy:', null_accuracy)

# Manual calculation of null accuracy by always predicting the majority class
#print('Manual null accuracy:',(1208 / (1208 + 185)))

#print confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_class)
#print(confusion_matrix)

# print message text for the false positives (ham incorrectly classified as spam)
# X_test[(y_pred_class==1) & (y_test==0)]
print(X_test[y_pred_class > y_test])

# print message text for the false negatives (spam incorrectly classified as ham)
#print(X_test[(y_pred_class==0) & (y_test==1)])
print(X_test[y_pred_class < y_test])

