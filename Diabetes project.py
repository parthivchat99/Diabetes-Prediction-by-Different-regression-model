#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dbts=pd.read_csv('diabetes.csv')
dbts


# In[5]:


dbts.info()


# In[6]:


dbts.isnull().sum()


# In[7]:


dbts.describe()


# In[8]:


plt.figure(figsize=(10,8))
sns.heatmap(dbts.corr(),annot=True,fmt=".3f",cmap="YlGnBu")
plt.title("Correlation Heatmap")


# In[9]:


plt.figure(figsize=(10,8))
kde=sns.kdeplot(dbts["Pregnancies"][dbts["Outcome"]==1],color="Blue",fill=True)
kde=sns.kdeplot(dbts["Pregnancies"][dbts["Outcome"]==0],color="Green",fill=True)
kde.set_xlabel("Pregnancies")
kde.set_ylabel("Density")
kde.legend("Positive","Negative")
plt.title("Distribution of Pregnancies with Outcome")


# In[10]:


plt.figure(figsize=(10,8))
kde=sns.kdeplot(dbts["Glucose"][dbts["Outcome"]==1],color="Red",fill=True)
kde=sns.kdeplot(dbts["Glucose"][dbts["Outcome"]==0],color="Yellow",fill=True)
kde.set_xlabel("Glucose")
kde.set_ylabel("Density")
kde.legend("Positive","Negative")
plt.title("Distribution of Glucose with Outcome")


# In[11]:


X=dbts.drop(columns='Outcome',axis=1)
Y=dbts['Outcome']
print(X)
print(Y)


# In[12]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)


# In[14]:


std_data=scaler.transform(X)
print(std_data)
X=std_data


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.35,stratify=Y,random_state=42)


# In[18]:


print(X.shape,X_train.shape,X_test.shape)
print(Y.shape,Y_train.shape,Y_test.shape)


# In[19]:


from sklearn import svm
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)


# In[22]:


from sklearn.metrics import accuracy_score
X_train_pred=classifier.predict(X_train)
train_data_accu=accuracy_score(X_train_pred,Y_train)
print("Accuracy score of training data: ",train_data_accu)
X_test_pred=classifier.predict(X_test)
test_data_accu=accuracy_score(X_test_pred,Y_test)
print("Accuracy score of testing data: ",test_data_accu)


# In[23]:


import numpy as np
ip_data=(4,211,92,3,0,37.6,0.181,30)
data_np_arr=np.asarray(ip_data)
ip_data_reshape=data_np_arr.reshape(1,-1)
std_ip=scaler.transform(ip_data_reshape)
print(std_ip)

prediction=classifier.predict(std_ip)


if prediction[0]==0:
  print("NON_DIABETIC")
else:
  print("DIABETIC")


# In[24]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=0)
classifier.fit(X_train, Y_train)


# In[25]:


X_train_pred=classifier.predict(X_train)
train_data_accu=accuracy_score(X_train_pred,Y_train)
print("Accuracy score of training data: ",train_data_accu)


# In[26]:


X_test_pred=classifier.predict(X_test)
test_data_accu=accuracy_score(X_test_pred,Y_test)
print("Accuracy score of testing data: ",test_data_accu)


# In[27]:


import numpy as np
ip_data=(4,211,92,3,0,37.6,0.181,30)
data_np_arr=np.asarray(ip_data)
ip_data_reshape=data_np_arr.reshape(1,-1)
std_ip=scaler.transform(ip_data_reshape)
print(std_ip)

prediction=classifier.predict(std_ip)


if prediction[0]==0:
  print("NON_DIABETIC")
else:
  print("DIABETIC")


# In[ ]:




