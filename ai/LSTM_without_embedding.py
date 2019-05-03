# Introduction:
	'''This task is a part of AI-hackathon conducted at Round Rock dell office to build a sentiment analysis model. Used LSTM framework with dropout on the reviews data only.
	Help is taken from Keras documentation'''
 
# Objective:
	'''The objective is to build a multiclass predictive model which can classify the review in the following ratings:

	* negative
	* somewhat negative
	* neutral
	* somewhat positive
	* positive '''
	
# download packages, In case they are not installed

!pip3 install keras
!pip3 install nltk
!pip3 install tensorflow
!pip3 install sklearn
!pip3 install re
!pip3 install nltk

#import packages

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))

#reading data
train = pd.read_csv("trainingData.csv")
test1 = pd.read_csv("testB_dell_reviews.csv")
test2 = pd.read_csv("test1_generic_reviews.csv")

# Renaming and dropping the unnecessary column
for df in [train,test1,test2]:
  df.rename(columns={'ReviewText':'Review'},inplace = True)
for df in [train,test1,test2]:
  df.drop(['Unnamed: 0'],axis=1,inplace = True)
  train = train[['Review','Rating']]
# filling null
for df in [train,test1,test2]:
  df.fillna('Unknown',inplace = True)
  
# Cleaning data
df = pd.concat([train.iloc[:,0:1],test1.iloc[:,0:1],test2.iloc[:,0:1]])
df = df.reset_index(drop=True)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    try:
      text = text.lower() # lowercase text
      text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
      text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
      text = text.replace('x', '')
#     text = re.sub(r'\W+', '', text)
      text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
      return text
    except:
      return "Unknown"
df['Review'] = df['Review'].apply(clean_text)
df['Review'] = df['Review'].str.replace('\d+', '')

# embedding
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250 # check for from text
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Review'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# tokenization of dataset
X = tokenizer.texts_to_sequences(df.iloc[0:len(train),]['Review'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

X_test1 = tokenizer.texts_to_sequences(df.iloc[len(train):len(train)+len(test1),]['Review'].values)
X_test1 = pad_sequences(X_test1, maxlen=MAX_SEQUENCE_LENGTH)

X_test2 = tokenizer.texts_to_sequences(df.iloc[len(train)+len(test1):,]['Review'].values)
X_test2 = pad_sequences(X_test2, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(train['Rating']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 41)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# Creating the model framework
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Running the model
epochs = 5
batch_size = 256
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

# prediction on test sets
pred_test1 = model.predict(X_test1)
pred_test2 = model.predict(X_test2)

# converting prediction to rating class based on maximum probability value
pred_test1_df = pd.DataFrame(data = pred_test1,columns=[1,2,3,4,5])
pred_test2_df = pd.DataFrame(data = pred_test2,columns=[1,2,3,4,5])

# adding predictions back to test dataset
test1['Rating']= pred_test1_df.idxmax(axis=1)
test2['Rating']= pred_test2_df.idxmax(axis=1)

# Saving results to file
test1.to_csv("TestB.csv",index = False)
test2.to_csv("Test1.csv",index = False)
