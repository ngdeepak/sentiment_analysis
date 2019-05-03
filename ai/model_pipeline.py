
# coding: utf-8

# In[13]:

# Fit LogisticRegression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
score


# In[14]:


o=classifier.score(X_train, y_train)


# In[ ]:

# Fit SVM 

from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, y_test)
print(accuracy)


# In[4]:

# Fit NN

import pickle
from scipy.sparse import hstack
f1 = open("nmf_features.pickle","rb")
f2 = open("tfidf.pickle","rb")
f3 = open("metaFeatures_new.pickle","rb")
b1 = pickle.load(f1)
b2 = pickle.load(f3)
b3 = pickle.load(f3)
temp= hstack((b1, b2))
temp1= hstack((temp, b3))
target = pd.read_csv("targetVariable.csv")


# In[12]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(temp2,target['Rating'], test_size=0.2, random_state=41)


# In[ ]:


from keras.models import Sequential
from keras import layers
input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                  epochs=100,
                  verbose=False,
                  validation_data=(X_test, y_test),
                  batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[5]:




