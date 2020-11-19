#!/usr/bin/env python
# coding: utf-8

# # Supervised Logistic Regression on Iris dataset

# #### Importing python libraries and modules of numpy and pandas

# In[39]:


# Importing the libraries
import pandas as pd
import numpy as np

# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Import scikit-Learn module for the algorithm/modeL: Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split

# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import scikit-Learn module classification report to later use for information about how the system try to classify / lable each record
from sklearn.metrics import classification_report


# #### Loading the dataset - Iris.csv

# * Title: Iris plants dataset
# * Number of instances: 150, 50 in each of three classes
# * Number of preditors: 4 numeric, predictive attributes and the class

# In[21]:


# Specify location of the dataset
filename = 'iris.csv'

# Load the data into a Pandas DataFrame
df = pd.read_csv(filename)


# #### Preprocessing the dataset

# * The following columns cannot contain 0 (zero) values.
# * i.e., 0 values are invalid in these columns
# * It they exist, we need to mark them as missing value or numpy.NaN

# 1. Set The results shows
# 2. Get the first five records 
# 3. Get the summary statistics of the numeric variables/attributes of the dataset

# In[22]:


# mark zero values as missing or NaN
df[[ 'SepalLengthCm' , 'SepalWidthCm' , 'PetalLengthCm ' ,'PetalWidthCm' ]] = df[['SepalLengthCm' , 'SepalWidthCm' ,'PetalLengthCm' , 'PetalWidthCm' ]].replace(0,np.NaN)

# count the number of NaN values in each column
print (df.isnull().sum())


# #### Performing EDA on the dataset

# * for each numberic variable/attribute of the dataset (VIP NOTES: The first variable ID is also plotted.
# * 5 numberic variables --> at least 5 plots --> Layout (2, 3): 2 rows, each row with 3 plots), boxplot, and scatter plot matrix

# In[23]:


# get the dimensions or shape of the dataset
# i.e. number of records / rows X number of variables / columns
print(df.shape)


# In[24]:


#get the data types of all the variables / attributes in the data set
print(df.dtypes)


# In[25]:


#return the first five records / rows of the data set
print(df.head(5))


# In[26]:


#return the summary statistics of the numeric variables / attributes in the data set
print(df.describe())


# In[27]:


#class distribution i.e. how many records are in each class
print(df.groupby('Species').size())


# In[28]:


#plot histogram of each numeric variable / attribute in the data set
df.hist(figsize=(12, 8))
pyplot.show()


# In[29]:


# generate density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# In[30]:


# generate box plots of each numeric variable / attribute in the data set
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()


# In[31]:


# generate scatter plot matrix of each numeric variable / attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()


# #### Seperating dataset into Input & Output NumPy arrays

# * Training data Set (X) consist of the independent variables or predictors
# * Desired Output (Y) consist of the dependent variable or that which we are trying to predict

# In[32]:


# store dataframe values into a numpy array
array = df.values

# separate array into input and output by slicing
# for X(input) [:, 1:5] --> all the rows, columns from 1 - 5 (6 - 1)
# these are the independent variables or predictors
X = array[:,1:5]

# for Y(input) [:, 5] --> all the rows, column 5
# this is the value we are trying to predict
Y = array[:,5]


# #### Split Input / Output arrays into training / testing datasets

# In[33]:


# split the dataset --> training sub-dataset: 67%; test sub-dataset: 33%
test_size = 0.33

#selection of records to include in each data sub-dataset must be done randomly
seed = 7

#split the dataset (input and output) into training / test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# #### Building and training the model

# In[34]:


#build the model
model = LogisticRegression()

# train the model using the training sub-dataset
model.fit(X_train, Y_train)

#print the classification report
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)

print(report)


# * The precision is the ratio tp / (tp + fp) --> where tp is the number of true positives and fp the number of false positives, the precision represents the ability of the classifier not to label a positive sample as negative
# * The recall is the ratio tp / (tp + fn) --> where tp is the number of true positives and fn the number of false negatives, the recall represents the ability of the classifier to find all the positive samples
# * The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall --> where an F-beta score reaches its best value at 1 and worst score at 0
# * The F-beta score weights recall more than precision by a factor of beta, beta == 1.0 means recall and precision are equally important
# * The support is the number of occurrences of each class in y_true

# #### Score the accuracy of the model

# In[35]:


#score the accuracy leve
result = model.score(X_test, Y_test)

#printing out the results
print(("Accuracy: %.3f%%") % (result*100.0))


# #### Classify / Predict Model

# In[36]:


model.predict([[5.3, 3.0, 4.5, 1.5]])


# * So, the model predict that the flower type of the new record is Iris-virginica
# * Based on the model's accuracy score: there is 90% chance that this new record is a Iris-virginica

# #### Evaluate the model using the 10-fold cross-validation technique.

# In[38]:


# evaluate the algorythm
# specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10

# fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 7

# split the whole dataset into folds
'''In k-fold cross-validation, the original sample is randomly partitioned into k equal sized
subsamples. Of the k subsamples, a single subsample is retained as the validation data for
testing the model, and the remaining k − 1 subsamples are used as training data. The crossvalidation process is then repeated k times, with each of the k subsamples used exactly once as
the validation data. The k results can then be averaged to produce a single estimation. The
advantage of this method over repeated random sub-sampling is that all observations are used for
both training and validation, and each observation is used for validation exactly once'''
kfold = KFold(n_splits, random_state=seed)

# for logistic regression, we can use the accuracy level to evaluate the model / algorithm
scoring = 'accuracy'

# train the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# print the evaluationm results
# result: the average of all the results obtained from the K-fold cross validation
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# * Using the 10-fold cross-validation to evaluate the model / algorithm, the accuracy of this logistic regression
# model is 88%
# * Above, the model predict that the flower type of the new record is Iris-virginica.
# * Based on the model's accuracy score obtained from the model evaluation using 10-told cross-validation: There is 88% chance that this new record is an Iris-virginica

# In[ ]:




