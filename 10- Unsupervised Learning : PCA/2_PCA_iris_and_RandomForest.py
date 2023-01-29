
# coding: utf-8

# # Implementing PCA with Scikit-Learn
In this section we will implement PCA with the help of Python's Scikit-Learn library. We will follow the classic machine learning pipeline where we will first import libraries and dataset, perform exploratory data analysis and preprocessing, and finally train our models, make predictions and evaluate accuracies. The only additional step will be to perform PCA to find out optimal number of features before we train our models.
# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/

# # Importing Libraries

# In[161]:


import numpy as np
import pandas as pd


# # Importing Dataset

# In[162]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)


# In[163]:


#dataset.head()
#print(dataset)


# # Preprocessing

# In[164]:


X = dataset.drop('Class', 1)
#print(X)

y = dataset['Class']
#print(y)
target=y.unique()
print(target)


# In[165]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split # Depricated

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # PCA performs best with a normalized feature set

# In[166]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Training and Making Predictions with RandomForestClassifier

# In[167]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# fit means to fit the model to the data being provided. This is where the model "learns" from the data.
# 
# transform means to transform the data (produce model outputs) according to the fitted model.
# 
# fit_transform means to do both - Fit the model to the data, then transform the data according to the fitted model. Calling fit_transform is a convenience to avoid needing to call fit and transform sequentially on the same input.

# # Performance Evaluation : Without PCA

# In[168]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
#print(cm)

accuracy=accuracy_score(y_test, y_pred)
print(accuracy)


# # PCA performs best with a normalized feature set.
Applying PCA
-------------------------------------------------------
Performing PCA using Scikit-Learn is a two-step process:

1. Initialize the PCA class by passing the number of components to the constructor.
2. Call the fit and then transform methods by passing the feature set to these methods. 
   The transform method returns the specified number of principal components.PCA depends only upon the feature set and not the label data.
Therefore, PCA can be considered as an unsupervised machine learning technique.
# In[169]:


from sklearn.decomposition import PCA
pca = PCA()               # All components
#pca = PCA(n_components=1)  # 1 components
#pca = PCA(n_components=2) # 2 components
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#print(pca)
#print(X_train)

The PCA class contains explained_variance_ratio_
which returns the variance caused by each of the principal components.
# In[43]:


explained_variance = pca.explained_variance_ratio_
print(explained_variance)


# # Visualization

#  integer encode : Category to integer

# In[193]:


from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
#print(integer_encoded)


# In[191]:


plt.scatter(X_train[:,0], X_train[:,1], c = integer_encoded)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# # Training and Making Predictions with RandomForestClassifier 

# In[194]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# # Performance Evaluation : With PCA

# In[45]:


cm = confusion_matrix(y_test, y_pred)
#print(cm)

accuracy=accuracy_score(y_test, y_pred)
print(accuracy)


# # Discussion
Discussion
From the above experimentation we achieved optimal level of accuracy while significantly reducing the number of features in the dataset. We saw that accuracy achieved with only 1 principal component is equal to the accuracy achieved with will feature set i.e. 93.33%. It is also pertinent to mention that the accuracy of a classifier doesn't necessarily improve with increased number of principal components. From the results we can see that the accuracy achieved with one principal component (93.33%) was greater than the one achieved with two principal components (83.33%).
# # Extra code

# In[66]:


from sklearn.metrics import accuracy_score
y_pred = [0, 1, 2, 3]
y_true = [0, 1, 2, 4]
accuracy_score(y_true, y_pred)
accuracy_score(y_true, y_pred, normalize=False)


# str is probably the builtin
# str_ may be numpy.str_ 

# In[38]:


#print(y_pred)
#print(type(y_pred))
y_pred=[str(x) for x in y_pred]
#y_pred=np.array(y_pred)
#print(y_pred)
print(type(y_pred))
print(type(y_pred[0]))
#print(y_test)
#print(type(y_test))
print('-----------------------')
y_test=[str(x) for x in y_test]
#print(y_test)
#y_test=np.array(y_test)
#print(y_test)
print(type(y_test))
print(type(y_test[0]))

accuracy=accuracy_score(y_test, y_pred)

print(accuracy)

#print('Accuracy' + classifier.score(X_test, y_test))


# In[189]:


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
#print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


# In[188]:


from sklearn.preprocessing import LabelEncoder
values=['Male','Female','Street']
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer 
One_hot_encoder = OneHotEncoder()
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
One_hot_encoder = One_hot_encoder.fit_transform(integer_encoded)
#One_hot_encoder = np.array(columnTransformer.fit_transform(values), dtype = np.str) 
print(One_hot_encoder)

