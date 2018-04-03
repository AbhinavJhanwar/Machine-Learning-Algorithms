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
import chardet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures

url = "C:/Users/abhinav.jhanwar/Desktop/Datasets/housing.csv"

with open(url) as csvFile:
    reader = csv.reader(csvFile)
    names = next(reader)

# encoding error correction    
#with open(url, 'rb') as f:
#    result = chardet.detect(f.read())  # or readline if the file is large
    
#data = pd.read_csv(url, encoding=result['encoding'])
data = pd.read_csv(url)

#print(data.head())
#print(data.tail())
#print(data.shape)
#Average_Rooms_in_Neighborhood  Student_vs_Teacher_Ratio Percent_Lower_Class_home_owners  House_Price
# this produces pairs of scatterplot as shown
# use aspect= to control the size of the graphs
# use kind='reg' to plot linear regression on the graph
#for i in range(0,len(names)):
#    sns.pairplot(data, x_vars=names[i], y_vars=names[-1], size=7, aspect=0.7, kind='reg')
#plt.show()

# create a fitted model
#lm1 = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
# print the coefficients
#print(lm1.params)
# summary
#print(lm1.summary())

names = ['per_capita_crime_rate', 'proportion_of_residential_land_over_25,000_sq.ft.', 'proportion_of_non-retail_business_acres_per_town', 'average_number_of_rooms_per_dwelling', 'proportion_of_owner-occupied_units_built_prior_to_1940', 'weighted_distances_to_five_Boston_employment_centres', 'full-value_property-tax_rate_per_$10,000', 'pupil-teacher_ratio_by_town', '%_lower_status_population', 'Median_House_Price']

feature_cols = names[0:-1] #names will be replaced by features directly taken from user selection
X = data[feature_cols]
y = data[names[-1]] #names replaced by target taken from user selection

# selct best features
#X_new = SelectKBest(f_regression, k=4).fit_transform(X,y)

validation_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=1)

model = LinearRegression()
#preprocessing with polynomial features
poly = PolynomialFeatures(degree=2)
#degre=2 means for two features x,y = b0+b1x+b2x2+b3xy+b4y2+b5y
X_train_ = poly.fit_transform(X_train)
X_test_ = poly.fit_transform(X_test)
model.fit(X_train_, y_train)

# y = \beta_0 + \beta_1 \times TV + \beta_2 \times Radio + \beta_3 \times Newspaper
# print coefficient and intercept
# beta0
#print(model.intercept_)      
# beta1, beta2, beta3
print(model.coef_)     

# pair the feature names with the coefficients
zip(feature_cols, model.coef_)
#print(list(zip(feature_cols, model.coef_)))
y_pred = model.predict(X_test_)

# rmse
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# creating csv file
predic = pd.DataFrame(data = {'Predicted_Values':y_pred}, index=y_test.index)
data = pd.concat([X_test, y_test, predic], axis=1)
csv_write = pd.DataFrame(data = data)
csv_write.to_csv("C:/Users/abhinav.jhanwar/Desktop/housingPrediction.csv")



