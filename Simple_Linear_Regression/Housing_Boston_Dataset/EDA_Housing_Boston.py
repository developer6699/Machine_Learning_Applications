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

