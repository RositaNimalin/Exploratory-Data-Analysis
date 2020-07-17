#!/usr/bin/env python
# coding: utf-8

# # EXPLORATORY DATA ANALYSIS ON BREAST CANCER DATASET
# ### By Rosita Nimalin 
# 
# 
# 
# **Dataset Description:**
# - Diagnosis : M = Malignant, B = Benign
# - Radius : Mean of distances from center to points on the perimeter
# - Texture : Standard Deviation of gray-scale values
# - Perimeter 
# - Area
# - Smoothness = local variation in radius lengths
# - Compactness = Perimeter^2/(area-1.0)
# - Concavity = Severity of concave portions of the contour
# - concave points = number of concave portions of the contour
# - Symmetry
# 
# 

# # Loading Libraries and Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time


# In[2]:


BC_df = pd.read_csv('BreastCancer.csv')


# ## Diving into the dataset

# In[3]:


BC_df.shape #dimension of the dataset


# The Breast Cancer dataset has 569 rows and 33 columns.

# In[4]:


BC_df.head()  #displaying first 5 rows of the dataset


# From the result, we note that:
# 
# - id column - for unique identity of the row which irrelevant for the classification.
# - diagonosis feature is categorical(M = malignant, B = benign)

# In[5]:


BC_df.info()  #to know the data shape of the dataset


# From the given information of the dataset, it implies that the features are computed with the mean, standard error (SE) and worst of each image. 

# In[6]:


BC_df.iloc[:,-1].unique() #to know if there are any other values in the unnamed column


# In[7]:


#to drop the id, diagnosis, unnamed columns from the dataset

y = BC_df.diagnosis #target class

drop_cols = ['Unnamed: 32','id','diagnosis']

x = BC_df.drop(drop_cols, axis = 1)
x.head()


# ## Plotting Diagnosis Distribution

# In[8]:


ax = sns.countplot(y, label = 'Count') #histogram across the target class (diagnosis) which is categorical
B, M = y.value_counts()
print("Count of Benign Tumors:",B)

print("Count of Malign Tumors:",M)


# In[9]:


x.describe() #summary (mean, std, quartiles, min and max)


# Result: We need to normalise the data as we can see that the values across features varies alot.

# In[10]:


data = x
data_std = (data - data.mean()) / data.std()  #normalising the data
data.head()


# In[11]:


data = pd.concat([y, data_std.iloc[:,:10]],axis = 1) #extracted 10 features from the dataset

data.head()


# In[12]:


data = pd.melt(data, id_vars='diagnosis', var_name = 'features', value_name = 'value') #unpivoting the datatable

data


# In[13]:


#to plot violinpolt
plt.figure(figsize = (10,10))
sns.violinplot(x ='features', y = 'value', hue = 'diagnosis', data = data, split = True, inner='quart')
plt.xticks(rotation = 45) #xlabels are rotated to 45 degrees so that it does not overlap


# From the plot Malign and Benign for fractal_dimension_mean feature, the median almost same while the it differs for the other features. Thus, the feature cannot give good information for the classification.

# ## Violin Plot for next 10 features
# 

# In[14]:


data = pd.concat([y, data_std.iloc[:,10:20]],axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name = 'features', value_name = 'value')
plt.figure(figsize = (10,10))
sns.violinplot(x ='features', y = 'value', hue = 'diagnosis', data = data, inner='quart') # here the split is False
plt.xticks(rotation = 45) 



# ## Violin plot for the last 10 features

# In[15]:


data = pd.concat([y, data_std.iloc[:,20:30]],axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name = 'features', value_name = 'value')
plt.figure(figsize = (10,10))
sns.violinplot(x ='features', y = 'value', hue = 'diagnosis', data = data, split = True, inner='quart')
plt.xticks(rotation = 45) #xlabels are rotated to 45 degrees so that it does not overlap


# From the plot we see that the features **concavity_worst** and **concave points_worst** looks correlated

# In[16]:


sns.boxplot(x = 'features', y = 'value',hue = 'diagnosis',data = data)
plt.xticks(rotation = 90)


# To find the correlation between the **concavity_worst** and **concave points_worst** using the jointplot from seaborn library

# In[17]:


sns.jointplot(x.loc[:,'concavity_worst'],
            x.loc[:,'concave points_worst'], kind = 'regg', color = 'blue')


# **Result**: The two variables are highly correlated

# ### Using Swarmplot to view the features

# In[18]:


plt.figure(figsize = (18,18))

plt.subplot(3,1,1)
data = pd.concat([y, data_std.iloc[:,:10]],axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name = 'features', value_name = 'value')
#plt.figure(figsize = (10,10))
sns.swarmplot(x ='features', y = 'value', hue = 'diagnosis', data = data)
plt.xticks(rotation = 45)


plt.subplot(3,1,2)
data1 = pd.concat([y, data_std.iloc[:,10:20]],axis = 1)
data1 = pd.melt(data1, id_vars='diagnosis', var_name = 'features', value_name = 'value')
#plt.figure(figsize = (10,10))
sns.swarmplot(x ='features', y = 'value', hue = 'diagnosis', data = data1)
plt.xticks(rotation = 45)

plt.subplot(3,1,3)
data = pd.concat([y, data_std.iloc[:,20:30]],axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name = 'features', value_name = 'value')
#plt.figure(figsize = (10,10))
sns.swarmplot(x ='features', y = 'value', hue = 'diagnosis', data = data)
plt.xticks(rotation = 45)



# In[20]:


f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(x.corr(), annot=True, linewidth =.5, fmt ='.1f', ax = ax)

