
# coding: utf-8

# In[38]:

#Trying different models

import pandas as pd
import pickle
from scipy.sparse import hstack

# We converted all  the feature set to sparse matrix

f1 = open("nmf_features.pickle","rb")
f2 = open("tfidf.pickle","rb")
f3 = open("metaFeaturesV2.pickle","rb")
b1= pickle.load(f1)
b2 = pickle.load(f2)
b3 = pickle.load(f3)
#temp1= hstack((b1, b2))
temp2= hstack((b1, b2))
target = pd.read_csv("targetVariable.csv")


# In[39]:

# Train Set Split

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(temp2,target['Rating'], test_size=0.2, random_state=41)


# In[34]:


b3


# In[40]:

# Trying LogisticRegression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)


# In[41]:


score


# In[6]:


classifier.score(X_train, y_train)


# In[44]:


x1 = open("lr_new.pickle","wb")
pickle.dump(classifier, x1)


# In[8]:

# Trying Naive Bayes


from sklearn.naive_bayes import MultinomialNB

mlt = MultinomialNB()
mlt.fit(X_train,y_train)
mlt.score(X_test, y_test)


# In[9]:


mlt.score(X_train, y_train)


# In[10]:


x2 = open("nb.pickle","wb")
pickle.dump(classifier, x2)


# In[ ]:

# Trying NN


from keras.models import Sequential
from keras import layers
input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                  epochs=10,
                  verbose=False,
                  validation_data=(X_test, y_test),
                  batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[ ]:

# Trying SVC

from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, y_test)
print(accuracy)


# In[35]:


x=pd.read_csv('train.csv')


# In[36]:


x.head(5)


# In[37]:


len(x)

