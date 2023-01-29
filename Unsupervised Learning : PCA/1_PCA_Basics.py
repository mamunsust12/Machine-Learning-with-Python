
# coding: utf-8

# # Principal Component Analysis(PCA) : IRIS Dataset

# In[2]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from numpy import linalg as LA

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(y)
#In general a good idea is to scale the data
X = stats.zscore(X)

pca = PCA()
x_new = pca.fit_transform(X)
#print(x_new)
plt.scatter(x_new[:,0], x_new[:,1], c = y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# # Principal Component Analysis(PCA) : 
# breast_cancer dataset

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[5]:


cancer=load_breast_cancer()


# In[6]:


cancer.keys()


# In[10]:


#cancer.feature_names


# In[ ]:


#print(cancer['DESCR'])


# In[8]:


#cancer.target


# In[11]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[12]:


df.head(5)


# In[13]:


from sklearn.preprocessing import MinMaxScaler


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


scaler=StandardScaler()
scaler.fit(df)


# In[16]:


scaled_data=scaler.transform(df)


# In[17]:


scaled_data


# In[19]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)


# In[20]:


scaled_data.shape


# In[21]:


x_pca.shape


# In[22]:


scaled_data


# In[23]:


x_pca


# In[25]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.show()

