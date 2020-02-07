#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[73]:


cc_apps = pd.read_csv('/CSV/Project/crx.csv',header=None)
cc_apps.head()


# ## Inspecting the applications

# In[74]:


cc_apps.describe()


# In[75]:


columns = ['Male','Age','Debt','Married','BankCustomer','EducationLevel',           'Ethnicity','YearsEmployed','PriorDefault','Employed','CreditScore','DriversLicense',        'Citizen','ZipCode','Income','Approved']


# In[76]:


cc_apps.columns = columns


# In[77]:


cc_apps.info()


# In[78]:


cc_apps.tail(20)


# ## Handling the missing values and wrong value type

# In[79]:


# Replace the '?'s with NaN
cc_apps = cc_apps.replace('?',np.nan)


# In[80]:


# Change the value type of Age to float
cc_apps['Age'] =cc_apps['Age'].astype('float')


# In[81]:


cc_apps.info()


# In[82]:


# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(),inplace=True)


# In[83]:


# Count the number of NaNs in the dataset to verify
cc_apps.isnull().sum()


# In[84]:


# Impute non_numeric missing valus with the most frequent value
for col in cc_apps.columns:
    if cc_apps[col].dtype =='object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])
# Count the number of NaNs in the dataset and print the counts to verify
cc_apps.isnull().sum()


# ## Processing the data

# In[85]:


from sklearn.preprocessing import LabelEncoder


# In[87]:


# Convert the non-numeric data into numeric for Maching learning model
le = LabelEncoder()
for col in cc_apps.columns:
    if cc_apps[col].dtype == 'object':
        cc_apps[col] = le.fit_transform(cc_apps[col])
cc_apps.head()


# In[88]:


# Split the data into train and test sets
from sklearn.model_selection import train_test_split

#Drop the features DrivesLicense and ZipCode and convert the DataFrame to a NumPy array
cc_apps = cc_apps.drop(['DriversLicense','ZipCode'],axis=1)
X = cc_apps.drop(['Approved'],axis='columns').values
y = cc_apps['Approved'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[41]:


# Scale the feature values to a uniform range
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
rescaledX_train = Scaler.fit_transform(X_train)
rescaledX_test = Scaler.fit_transform(X_test)


# ## Fitting a logistic regression model to the train set

# In[89]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
logreg = LogisticRegression()
logreg.fit(rescaledX_train,y_train)


# ## Making predictions and evaluating performance

# In[92]:


y_pred = logreg.predict(rescaledX_test)
print('accuracy',logreg.score(rescaledX_test,y_test))
print(confusion_matrix(y_test,y_pred))


# In[93]:


print(classification_report(y_test,y_pred))


# ## Grid searching and making the model perform better

# In[94]:


from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]
params = {'tol':tol,'max_iter':max_iter}


# ## Finding the best performing model

# In[95]:


cv = GridSearchCV(logreg,params,cv=5)
rescaledX = Scaler.fit_transform(X)
cv.fit(rescaledX,y)
print(cv.best_score_,cv.best_params_)

