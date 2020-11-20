#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: Supervised Classification KNN

# In[1]:


# Import Python Libraries: NumPy and Pandas
import pandas as pd
import numpy as np

# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Import scikit-Learn module for the algorithm/modeL: Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split

# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import scikit-Learn module classification report to later use for information about how the
# system try to classify / lable each record
from sklearn.metrics import classification_report


# ### Loading the Iris.csv dataset 

# In[3]:


# Specify location of the dataset
filename = 'iris.csv'

# Load the data into a Pandas DataFrame
df = pd.read_csv(filename)

df.head()


# # Preprocessing the dataset

# In[4]:


# mark zero values as missing or NaN
df[[ 'SepalLengthCm' , 'SepalWidthCm' , 'PetalLengthCm ' ,'PetalWidthCm' ]] = df[['SepalLengthCm' , 'SepalWidthCm' ,'PetalLengthCm' , 'PetalWidthCm' ]].replace(0,np.NaN)


# In[5]:


# count the number of NaN values in each column
print (df.isnull().sum())


# In[6]:


# get the dimensions or shape of the dataset
# i.e. number of records / rows X number of variables / columns
print(df.shape)


# In[7]:


#get the data types of all the variables / attributes in the data set
print(df.dtypes)


# In[8]:


#return the first five records / rows of the data set
print(df.head(5))


# In[9]:


#return the summary statistics of the numeric variables / attributes in the data set
print(df.describe())


# In[10]:


#class distribution i.e. how many records are in each class
print(df.groupby('Species').size())


# In[11]:


#plot histogram of each numeric variable / attribute in the data set
df.hist(figsize=(12, 8))
pyplot.show()


# In[12]:


# generate density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# In[13]:


# generate box plots of each numeric variable / attribute in the data set
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()


# In[14]:


# generate scatter plot matrix of each numeric variable / attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()


# ### Separating the dataset into Input & Output NumPy arrays

# In[15]:


# store dataframe values into a numpy array
array = df.values


# In[16]:


# separate array into input and output by slicing
# for X(input) [:, 1:5] --> all the rows, columns from 1 - 4 (5 - 1)
# these are the independent variables or predictors
X = array[:,1:5]


# In[17]:


# for Y(input) [:, 5] --> all the rows, column 5
# this is the value we are trying to predict
Y = array[:,5]


# ### Split Input/Output Arrays into Training/Testing Datasets

# In[18]:


# split the dataset --> training sub-dataset: 67%; test sub-dataset: 33%
test_size = 0.33


# In[19]:


#selection of records to include in each data sub-dataset must be done randomly
seed = 7


# In[20]:


#split the dataset (input and output) into training / test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# ### Building / Training the model

# In[21]:


#build the model
model = KNeighborsClassifier()


# In[22]:


# train the model using the training sub-dataset
model.fit(X_train, Y_train)


# In[23]:


#print the classification report
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)


# ### Score the accuracy of the model

# In[24]:


#score the accuracy level
result = model.score(X_test, Y_test)


# In[25]:


#print out the results
print(("Accuracy: %.3f%%") % (result*100.0))


# ### Classifying / Predicting the model

# In[27]:


model.predict([[5.3, 3.0, 4.5, 1.5]])


# ### Evaluating the model using the 10-fold cross-validation technique.

# In[28]:


# evaluate the algorythm
# specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10


# In[29]:


# fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 7


# In[30]:


# split the whole dataset into folds
# In k-fold cross-validation, the original sample is randomly partitioned into k equal sized
# subsamples. Of the k subsamples, a single subsample is retained as the validation data for
# testing the model, and the remaining k − 1 subsamples are used as training data. The crossvalidation
# process is then repeated k times, with each of the k subsamples used exactly once as
# the validation data. The k results can then be averaged to produce a single estimation. The
# advantage of this method over repeated random sub-sampling is that all observations are used for
# both training and validation, and each observation is used for validation exactly once
kfold = KFold(n_splits, random_state=seed)


# In[31]:


# for logistic regression, we can use the accuracy level to evaluate the model / algorithm
scoring = 'accuracy'


# In[32]:


# train the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)


# In[33]:


# print the evaluationm results
# result: the average of all the results obtained from the K-fold cross validation
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:




