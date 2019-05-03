
# coding: utf-8

# In[89]:

# Trying to retrieve predictions

import pickle
from scipy.sparse import hstack
f1 = open("test_1_nmf.pickle","rb")
f2 = open("test_1_meta.pickle","rb")
f3 = open("test_1_vect.pickle","rb")
b1 = pickle.load(f1)
b2 = pickle.load(f2)
b3 = pickle.load(f3)
#temp= hstack((b1, b2))
temp1= hstack((b1, b3))


# In[78]:


b1


# In[91]:

# Fit LR on the data


tp= open("lr_new.pickle","rb")
model= pickle.load(tp)


# In[92]:


y_pred=model.predict(temp1)


# In[93]:


y_pred


# In[94]:


import pandas as pd
df= pd.read_csv('test1.csv')


# In[95]:


df['Rating']=y_pred


# In[96]:


df=df[['ReviewText','Rating']]


# In[98]:


df.to_csv('baseline_dataset1.csv',index=False)


# In[99]:


df.groupby('Rating').agg('count')


# In[65]:


temp=pd.read_csv('baseline_dataset1.csv')


# In[76]:


df.groupby('Rating').agg('count')

