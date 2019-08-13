#!/usr/bin/env python
# coding: utf-8

# In[157]:


get_ipython().magic('matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[158]:


data_matrix = np.zeros((1000,2))

count =0 
x = 0

for i in range(1000):
    data_matrix[i][0] = x*0.7 
    data_matrix[i][1] = data_matrix[i][0] + 50
    x = x + 0.01
    
data = pd.DataFrame(matrix, columns=['x','y'])


# In[159]:


plt.scatter(data.x, data.y, s=0.3, c='b')


# In[167]:


mu, sigma = 0, 50 
noise_matrix = np.random.normal(mu, sigma, [1000,2])


# In[173]:


signal = data + noise_matrix


# In[174]:


plt.scatter(signal.x, signal.y,s =0.7,c='black')


# In[196]:


lambdav, v = np.linalg.eig(signal.cov())
print("Eigen values ", lambdav)
print("Eigen vectors ", v)
mu = np.mean(signal)
print(mu)

v1 = np.zeros((1000,2))
v2 = np.zeros((1000,2))

x = 0.0
for i in range(1000):
    v1[i][0] = x*v[0][0] + mu[0]
    v1[i][1] = x*v[0][1] + mu[1]
    v2[i][0] = x*v[1][0] + mu[0]
    v2[i][1] = x*v[1][1] + mu[1]
    x += 0.15


# In[197]:


plt.scatter(signal.x, signal.y,s =0.7, c='black')
plt.plot(v1[:,0], v1[:,1], color='red', label = 'Eigen v1', linewidth=2.0)
plt.plot(v2[:,0], v2[:,1], color='green', label = 'Eigen v2', linewidth=2.0)

