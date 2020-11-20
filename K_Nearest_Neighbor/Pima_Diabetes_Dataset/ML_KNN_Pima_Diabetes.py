#!/usr/bin/env python
# coding: utf-8

# # ML - K Nearest Neighbor on Pima Diabetes dataset

# In[1]:


# Import Python Libraries: NumPy and Pandas
import pandas as pd
import numpy as np

# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

# Import scikit-Learn module for the algorithm/modeL: Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Import scikit-Learn module classification report to later use for information about how the
# system try to classify / lable each record
from sklearn.metrics import classification_report


# ### Load the dataset: pima_diabetes.csv

# In[2]:


df = pd.read_csv('pima_diabetes.csv')


# In[3]:


df.head()


# # Preprocess Dataset

# In[4]:


# Count the number of NaN values in each column
print (df.isnull().sum())


# In[5]:


# Getting to know the dimensions or shape of the dataset, number of records, rows X number of variables, columns
print(df.shape)


# In[6]:


# Getting to know the data types of all the variables / attributes in the data set
print(df.dtypes)


# In[7]:


# Return the first five records, rows of the data set
print(df.head(5))


# In[8]:


# Return the summary statistics of the numeric variables / attributes in the data set
print(df.describe())


# In[9]:


# Class distribution i.e; how many records are in each class
print(df.groupby('class').size())


# In[10]:


# Plotting histogram of each numeric variable / attribute in the data set
df.hist(figsize=(12, 8))
plt.show()


# In[11]:


# Generating density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
plt.show()


# In[12]:


# Generating box plots of each numeric variable / attribute in the data set
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
plt.show()


# In[13]:


# Generating scatter plot matrix of each numeric variable / attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
plt.show()


# ### Separate Dataset into Input & Output NumPy arrays

# In[14]:


# Storing dataframe values into a numpy array
array = df.values

X = array[:,0:-1]
Y = array[:,-1]


# ### Split Input/Output Arrays into Training/Testing Datasets

# In[22]:


# Spliting the dataset --> training sub-dataset: 67%; test sub-dataset: 33%
test_size = 0.33

# Selection of records to include in each data sub-dataset must be done randomly
seed = 7

# Spliting the dataset (input and output) into training / test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# ### Building / Training the Model

# In[23]:


# Building the model
model = KNeighborsClassifier()


# In[24]:


# Training the model using the training sub-dataset
model.fit(X_train, Y_train)


# In[25]:


# Printing the classification report
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)


# ### Score the accuracy of the model

# In[26]:


# Scoring the accuracy level
result = model.score(X_test, Y_test)


# In[27]:


# Printing out the results
print(("Accuracy: %.3f%%") % (result*100.0))


# ### 1st Prediction

# In[28]:


model.predict([[4,121,69,20,80,32,0.472,33]])


# ### 2nd Prediction

# In[30]:


model.predict([[6,140,80,32,127,37,0.626,41]])


# ### Evaluating the model using the 10-fold cross-validation technique.

# In[31]:


# Evaluating the algorithm by specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10

# Fix the random seed must use the same seed value so that the same subsets can be obtained
seed = 7

# Spliting the whole dataset into k equal sized subsamples. 
kfold = KFold(n_splits, random_state=seed)

# For KNN, we can use the accuracy level to evaluate the model / algorithm
scoring = 'accuracy'

# Training the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)


# In[32]:


# Printing the evaluationm results of the average of all the results obtained from the K-fold cross validation
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:




