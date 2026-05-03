#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model training logic for water quality prediction
# In[36]:


df=pd.read_csv(r"D:\Brave Downloads\water_potability.csv")
df.head()


# In[37]:


df.shape


# In[38]:


df.isnull().sum()


# In[39]:


df.info()


# In[40]:


df.describe()


# In[41]:


df.fillna(df.mean(), inplace=True)
df.head()


# In[42]:


df.describe()


# In[43]:


df.info()


# In[44]:


df.Potability.value_counts()


# In[45]:


df.Potability.value_counts().plot(kind="bar", color=["brown", "salmon"])
plt.show()


# In[46]:


sns.distplot(df['ph'])


# In[47]:


df.hist(figsize=(14,14))
plt.show()


# In[48]:


sns.heatmap(df.corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[49]:


df.boxplot(figsize=(14,7))


# In[50]:


df['Solids'].describe()


# In[51]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[52]:


x = df.drop('Potability',axis=1)
y=df['Potability']
y


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size= 0.2, random_state=101,shuffle=True)


# In[54]:


X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[56]:


classifier=RandomForestClassifier(n_estimators=269,max_depth=14,random_state=69)


# In[57]:


classifier.fit(X_train_scaled,Y_train)


# In[58]:


y_pred=classifier.predict(X_test_scaled)


# In[59]:


from sklearn.metrics import accuracy_score


# In[60]:


accuracy_score(y_pred,Y_test)


# In[68]:


import pickle
import streamlit as st
pickle.dump(classifier, open('classifier.pkl', 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))


# In[62]:


import os
import pickle

# Check the current working directory
current_dir = os.getcwd()
print("Current directory:", current_dir)

# Construct the absolute path to the pickle file
pickle_file = os.path.join(current_dir, "classifier.pkl")

# Check if the file exists
if os.path.exists(pickle_file):
    # File exists, load the classifier
    with open(pickle_file, "rb") as f:
        classifier = pickle.load(f)
    print("Classifier loaded successfully.")
else:
    # File does not exist, print an error message
    print("Error: 'classifier.pkl' file not found.")

# Now you can use the 'classifier' object for further processing


# In[69]:


import os

# Get the current directory
current_dir = os.getcwd()

# Construct the absolute path to the pickle file
pickle_file = os.path.join(current_dir, "classifier.pkl")
scaler_file = os.path.join(current_dir, "scaler.pkl")

# Print the directory of the pickle file
print("Directory of 'classifier.pkl':", pickle_file)
print("Directory of 'classifier.pkl':", scaler_file)


# In[33]:





# In[72]:


X_test


# In[65]:


y_pred


# In[66]:


Y_test


# In[ ]:




