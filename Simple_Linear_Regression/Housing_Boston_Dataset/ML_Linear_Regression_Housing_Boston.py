#!/usr/bin/env python
# coding: utf-8

# ### Visualizing data with matplotlib - stateful approach with pyplot

# #### Importing the libraries

# In[2]:


# Importing matplotlib
import matplotlib.pyplot as plt


# #### Declaring two lists named x and y

# In[3]:


# Declaring the lists
x = [-3, 5 , 7]
y = [10, 2, 5]


# ##### Now we are illustrating the above lists by plotting a graph

# In[6]:


plt.figure(figsize=(15,3))
plt.plot(x,y)
plt.xlim(0,10)
plt.ylim(-3,8)
plt.xlabel('X coordinate axis')
plt.ylabel('Y coordinate axis')
plt.title('Line plot of the list')


# #### Stateless visualization
# ##### Visualization with class axes: Stateless (OO Approach)

# In[7]:


# Importing matplotlib using pyplot
import matplotlib.pyplot as plt

# Calling pyplot subplots() for creating a figure object, which creates one axes
fig, ax = plt.subplots(figsize=(15,3))

ax.plot(x,y)
ax.set_xlim(0,10)
ax.set_ylim(-3,8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Line plot for the axes')

plt.show()


# #### Import & Load dataset

# In[8]:


# Here we are importing all the necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from pandas import DataFrame, read_csv


# In[10]:


# Here we are loading the dataset into a pandas datafrme
df = pd.read_csv('Iris.csv')

# To view the sample dataset
df.head()


# #### Data visualization with pandas & matplotlib

# ##### Univariate data visualization

# ###### 1. Using histograms

# In[12]:


# Visualizations using histograms
df.hist(figsize=(12,8))
plt.show


# ###### 2. Using density plots

# In[15]:


# Visualizations using density plots
df.plot(kind='density', subplots=True, layout=(2,3), sharex=False, legend=True, fontsize=1, figsize=(12,16))
plt.show()


# ###### 3. Using Box or  Whisker plots

# In[16]:


# Visualizations using box or whisker plots
df.plot(kind='box', subplots=True, layout=(2,3), sharex=False, legend=True, fontsize=1, figsize=(12,8))
plt.show()


# ##### Multivariate data visualization

# ###### 1. Scatter Matrix plot

# In[17]:


# Visualizations using scatter matrix plot
scatter_matrix(df, alpha=0.8, figsize=(9,9))
plt.show()


# # ----------------------------------------------------------------------------------------------------------

# # Supervised Linear Regression

# ##### Import python libraries and modules

# In[19]:


# Importing libraries
import pandas as pd
import numpy as np

# Import scikit-Learn module for the algorithm/modeL: Linear Regression
from sklearn.linear_model import LinearRegression

# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split

# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ##### Loading dataset

# In[21]:


# Specify location of the dataset
filename = 'housing boston.csv'

# Specify the fields with their names
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
'LSTAT', 'MEDV']

# Load the data into a Pandas DataFrame
df = pd.read_csv(filename, names=names)

# VIP NOTES: 
# Extract a sub-dataset from the original one -- > dataframe: df2
df2 = df[['RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'MEDV']]


# #### Pre-processing the dataset

# ###### Cleaning data by finding and marking missing values

# In[22]:


# So as few columns cannot contain zero values.
# Mark zero values as missing or NaN. So there is no invalid value in any value of the original data frame 
df[['RM', 'PTRATIO', 'MEDV']] = df[['RM', 'PTRATIO', 'MEDV']].replace(0, np.NaN)

# count the number of NaN values in each
print(df.isnull().sum())


# ##### Performing the Exploratory data analysis on the dataset

# In[23]:


# Get the dimensions or Shape of the dataset, i.e. number of records/rows x number of variables/columns
print(df2.shape)


# In[24]:


# Get the data types of all variabLes/attributes of the data set, The results show
print(df2.dtypes)


# In[25]:


# Get several records/rows at the top of the dataset
# Get the first five records
print(df2.head(5))


# In[26]:


# Get the summary statistics of the numeric variables/attributes of the dataset
print(df2.describe())


# In[27]:


# Plot histrogram for each numeric
df2.hist(figsize=(12, 8))
pyplot.show()


# In[29]:


# Density plots
# IMPORTANT NOTES: 5 numeric variables -> at Least 5 plots -> Layout (2, 3): 2 rows, each row with 3 plots
df2.plot(kind='density', subplots=True, layout=(2, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# In[33]:


# Visualizations using box or whisker plots
df2.plot(kind='box', subplots=True, layout=(3,3), sharex=False, legend=True, fontsize=1, figsize=(12,8))
pyplt.show()


# In[34]:


# scatter plot matrix
scatter_matrix(df2, alpha=0.8, figsize=(15, 15))
pyplot.show()


# ##### Seperating dataset into input & output by using numpy arrays

# In[36]:


# Store dataframe values into a numpy array
array = df2.values
# separate array into input and output components by slicing
# For X (input)[:, 5] --> all the rows, columns from 0 - 4 (5 - 1)
X = array[:,0:5]
# For Y (output)[:, 5] --> all the rows, column index 5 (Last column)
Y = array[:,5]


# ##### We will be storing this dataset into multi dimensional array, y is dependent variable

# ##### We are splitting Input / Output arrays into Training / Testing datasets

# In[37]:


# Split the dataset --> training sub-dataset: 67%; test sub-dataset:
test_size = 0.33

# Selection of records to include in which sub-dataset must be done randomly
# use this seed for randomization
seed = 7

# Split the dataset (both input & outout) into training/testing datasets
# So the training set is 70% of dataset and test set is 30%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)


# ##### Now its time to build and train the model

# In[38]:


# Building the model
model = LinearRegression()
# Train the model using the training sub-dataset
model.fit(X_train, Y_train)
# Print out the coefficients and the intercept
# print intercept and coefficients
print (model.intercept_)
print (model.coef_)


# In[40]:


# If we want to print out the list of the coefficients with their correspondent variable name
# pair the feature names with the coefficients
names_2 = ['RBI', 'AGE','DIS','RAD', 'PTRATIO']
coeffs_zip = zip(names_2, model.coef_)

# Convert iterator into set
coeffs = set(coeffs_zip)

#print (coeffs)
for coef in coeffs:
    print (coef, "\n")


# In[41]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


# #### Calculating R-Squared

# In[42]:


R_squared = model.score(X_test, Y_test)
print(R_squared)


# #### Now as we have build and train the model, time for prediction

# In[43]:


model.predict([[6.0, 55, 5, 2, 16]])


# #### Evaluating, Validating algorithm, Model
# #### Using K-Fold Cross Validation

# In[45]:


# Evaluate the algorithm
# Specify the K-size
num_folds = 10

# Fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 7

# Split the whole data set into folds
kfold = KFold(n_splits=num_folds, random_state=seed)

# For Linear regression, we can use MSE (mean squared error) value
# to evaluate the model/algorithm
scoring = 'neg_mean_squared_error'

# Train the model and run K-foLd cross-validation to validate/evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

# Print out the evaluation results
# Result: the average of all the results obtained from the k-foLd cross-validation
print(results.mean())


# #### Conclusion: So as the training and evaluating done. We have used K-Fold to determine if the model is acceptable
# #### -31 avg of all error (mean of square errors) this value would traditionally be positive value, we can see that scikit reports as neg,
# #### the square root would be between 5 and 6

# In[ ]:




