#!/usr/bin/env python
# coding: utf-8

# # ML - Logistic Regression on Pima Diabetes dataset

# In[1]:


# Import python libraries: NumPy and Pandas and its classes
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv

# Import libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

#Import libraries for modeling
from sklearn.linear_model import LogisticRegression

# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split

# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *


# ### Loading Dataset

# In[2]:


df = pd.read_csv('pima_diabetes.csv')


# ### Performing Exploratory Data Analysis

# In[3]:


# Getting to know the dimensions or shape of the dataset, number of records, rows X number of variables, columns
print(df.shape)


# In[4]:


# Getting to know the data types of all the variables / attributes in the data set
print(df.dtypes)


# In[5]:


# Returning the first five records, rows of the data set
print(df.head(5))


# In[6]:


# Returning the summary statistics of the numeric variables, attributes in the data set
print(df.describe())


# In[7]:


# Plotting histogram of each numeric variable / attribute in the data set
df.hist(figsize=(12, 8))
plt.show()


# In[8]:


# Generating the density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
plt.show()


# In[9]:


# Generating the box plots of each numeric variable, attribute in the data set
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
plt.show()


# In[10]:


# Generating the scatter plot matrix of each numeric variable, attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
plt.show()


# ### Separate Dataset into Input & Output NumPy arrays

# In[11]:


# Storing dataframe values into a numpy array
array = df.values


# In[12]:


# Separating the array into input and output by slicing
X = array[:,0:-1]
Y = array[:,-1]


# ### Split Input/Output Arrays into Training/Testing Datasets

# In[13]:


# Spliting the dataset --> training sub-dataset: 67%; test sub-dataset: 33%
test_size = 0.33

# Selecting of records to include in each data sub-dataset must be done randomly
seed = 7

# Spliting the dataset (input and output) into training / test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=seed)


# ### Building / Training a model

# In[14]:


# Building the model
model = LogisticRegression()

# Training the model using the training sub-dataset
model.fit(X_train, Y_train)

predicted = model.predict(X_test)


# In[15]:


report = classification_report(Y_test, predicted)
print(report)


# ### Score the accuracy of the model

# In[16]:


# Scoring the accuracy level
result = model.score(X_test, Y_test)


# In[17]:


print(("Accuracy: %.3f%%") % (result*100.0))


# ### 1st Prediction

# In[18]:


model.predict([[4,121,69,20,80,32,0.472,33]])


# ### 2nd Prediction

# In[19]:


model.predict([[6,140,80,32,127,37,0.626,41]])


# ### Evaluate the model using the 10-fold cross-validation technique.

# In[20]:


# Evaluating the algorythm, specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10

# Fixing the random seed must use the same seed value so that the same subsets can be obtained for each time the process is repeated
seed = 7

# Spliting the whole dataset into k equal sized subsamples. 
kfold = KFold(n_splits, random_state=seed)

# For logistic regression, we can use the accuracy level to evaluate the model / algorithm
scoring = 'accuracy'

# Training the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)


# In[21]:


# Printing the evaluationm results obtained from the K-fold
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:




