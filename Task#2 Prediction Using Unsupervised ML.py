#!/usr/bin/env python
# coding: utf-8

# ## <span style="color:blue"> Submitted By - SUMUNNA PAUL</span>
# 
# ## <span style="color:purple"> DATA SCIENCE AND ANALYTICS INTERN</span>
# 
# ## <span style="color:red"> THE SPARKS FOUNDATION INTERNSHIP PROGRAM JULY'21</span>
# 
# # Task:2 Prediction Using Unsupervised ML
# 
# In this case study, my task is to create a machine learning model which can predict the optimum number of cluster and represent it visually.
# 
# In below case study I will discuss some of the basics of K-Means Clustering.
# 
# ## Reading the data into python
# The data has one file "Iris.csv". This file contains 150 flower species data.
# 
# ## Data description
# The business meaning of each column in the data is as below
# 
# * <b>SepalLengthCm</b>: Length of Sepal in cm
# * <b>SepalWidthCm</b>: Width of Sepal in cm
# * <b>PetalLengthCm</b>: Length of Petal in cm
# * <b>PetalWidthCm</b>: Width of Petal in cm
# * <b>Species</b>: Species of the flower

# In[1]:


# Reading the dataset
import numpy as np
import pandas as pd
Iris=pd.read_csv(filepath_or_buffer="C:/Users/Pratik/Desktop/IVY ProSchool/Sparks Internship/Iris.csv",sep=',', encoding='latin-1')
print('Shape of data before removing duplicate data :',Iris.shape)
Iris=Iris.drop_duplicates()
print('Shape of data after removing duplicate data :',Iris.shape)
Iris.head(10)


# ## Basic Data Exploration
# Initial assessment of the data should be done to identify which columns are Quantitative, Categorical or Qualitative.

# In[2]:


Iris.head()


# In[3]:


Iris.info()


# In[4]:


Iris.describe(include='all')


# In[5]:


Iris.nunique()


# Based on the basic exploration above, you can now create a simple report of the data
# 
# * <b>Species</b>: Selected. Categorical.
# * <b>SepalLengthCm</b>: Selected. Continuous
# * <b>SepalWidthCm</b>: Selected. Continuous
# * <b>PetalLengthCm</b>: Selected. Continuous
# * <b>PetalWidthCm</b>: Selected. Continuous
# 
# ## Removing useless columns from the data

# In[6]:


Iris= Iris.drop(['Id'],axis=1)


# In[8]:


Iris.shape


# ## Missing values treatment
# Missing values are treated for each column separately.
# If a column has more than 30% data missing, then missing value treatment cannot be done.

# In[9]:


Iris.isnull().sum()


# There is no value missing in this data.

# ## Normalization
# let's do the normalization excluding the categorical attribute. let's create the Normalization function.

# In[10]:


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return x


# In[11]:


#lets consider the given data set excluding the last column
Iris_norm=norm_func(Iris.iloc[:, :4])
Iris_norm


# ## Elbow Curve

# In[14]:


#lets import KMeans library
# Finding the optimum number of clusters for k-means classification
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in list(range(2,9)):
    kmeans= KMeans(n_clusters=i)
    kmeans.fit(Iris_norm)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(list(range(2,9)), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# You can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# 
# From this we choose the number of clusters as ** '3**'.
# 
# ## Graphical Representation

# In[25]:


#lets fit the model & create the label
model=KMeans(n_clusters=3)
model.fit(Iris_norm)
model.labels_
Label=pd.Series(model.labels_)
#create a new column named 'Cluster'
Iris['Cluster']=Label 
Iris.head()


# In[26]:


#Lets rearranging the column position
Iris=Iris.iloc[:,[5,0,1,2,3,4]]


# In[27]:


Iris.info()


# In[28]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
         
                max_iter = 300, n_init = 10, random_state = 0)
x= Iris.iloc[:,1:5].values
y_kmeans = kmeans.fit_predict(x)


# In[29]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




