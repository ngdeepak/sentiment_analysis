
# coding: utf-8

# In[39]:


import pandas as pd
import textblob
from sklearn import preprocessing
import numpy as np
from time import time
import scipy as sp
import sklearn
import sys
from nltk.corpus import stopwords
import nltk
from gensim.models import ldamodel
from sklearn.preprocessing import MinMaxScaler
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
import pickle
import numpy as np
from scipy import sparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
sid = SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')
import logging
logging.getLogger('').handlers = []
logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='features.log',
                            filemode='w')


# In[40]:


tp= open("data_tp.pickle","rb")
base= pickle.load(tp)


# In[41]:


# Generating Topic Features

logging.info("Starting NMF")
data = [' '.join(text) for text in base]
vectorizer = CountVectorizer(analyzer='word', max_features= 100000)
x_counts = vectorizer.fit_transform(data)
transformer = TfidfTransformer(smooth_idf=False)
x_tfidf = transformer.fit_transform(x_counts)
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
num_topics= 20
model = NMF(n_components=num_topics, init='nndsvd')
model.fit(xtfidf_norm)
logging.info("Ending NMF")


# In[19]:


def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i)] = words
    
    return pd.DataFrame(word_dict)


# In[ ]:


get_nmf_topics(model, 20)


# In[21]:


v_set= model.transform(vectorizer.transform(data))


# In[22]:


data_df= pd.DataFrame(v_set)


# In[27]:


data_df

meta= pd.read_csv('backup.csv')


# In[30]:


meta.head(5)
meta=meta[['wc_f1','uwc_f2','wc_f3','compound',
       'positive','negative','neutral','noun_count','verb_count',
        'adj_count','adv_count','pron_count','char_count','word_density']]
result = pd.concat([data_df, meta], axis=1, sort=False)


# In[34]:


features.to_csv('features_norm.csv')


# In[ ]:


scaler = MinMaxScaler()
features = pd.DataFrame(scaler.fit_transform(result))
features


# In[36]:


vec_features = sparse.csr_matrix(features)

pickle_tpp = open("metaFeaturesV2.pickle","wb")
pickle.dump(vec_features, pickle_tpp)


# In[7]:



pickle_tp = open("nmf_features.pickle","wb")
pickle.dump(topic_vec, pickle_tp)


# In[ ]:


# Sentiment Compound Score

df= pd.read_csv('cleaned_df.csv')
df['scores'] = df['ReviewText'].apply(lambda review: sid.polarity_scores(review))
df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['positive']  = df['scores'].apply(lambda score_dict: score_dict['pos'])
df['negative']  = df['scores'].apply(lambda score_dict: score_dict['neu'])
df['neutral']  = df['scores'].apply(lambda score_dict: score_dict['neg'])


# In[3]:


df= pd.read_csv('featureSet.csv')
df.head(5)


# In[4]:


df['char_count'] = df['ReviewText'].apply(len)
df['word_density'] = df['char_count'] / (df['wc_f1']+1)

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

df['noun_count']= df['ReviewText'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count']= df['ReviewText'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count']= df['ReviewText'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count']= df['ReviewText'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count']= df['ReviewText'].apply(lambda x: check_pos_tag(x, 'pron'))


# In[15]:


vec_features = sparse.csr_matrix(df)
scaler = MinMaxScaler()
features = scaler.fit_transform(df)
pickle_tpp = open("metaFeatures_new.pickle","wb")
pickle.dump(vec_features, pickle_tpp)


# In[13]:


df=df[['wc_f1','uwc_f2','wc_f3','compound',
       'positive','negative','neutral','noun_count','verb_count',
        'adj_count','adv_count','pron_count','char_count','word_density']]


# In[5]:


# TF-IDF Features
data = [' '.join(text) for text in base]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)


# In[37]:


with open('vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)


# In[6]:


pickle_tr = open("tfidf.pickle","wb")
pickle.dump(X, pickle_tr)


# In[22]:


temp=df.copy()
temp=temp[['wc_f1','uwc_f1','wc_f3','compound',
       'positive','negative','neutral','noun_count','verb_count',
        'adj_count','adv_count','pron_count','char_count','word_density']]

scaler = MinMaxScaler()
features = scaler.fit_transform(temp)
vec_features = sparse.csr_matrix(features)

pickle_tpp = open("metaFeatures.pickle","wb")
pickle.dump(vec_features, pickle_tpp)


# In[29]:


temp= hstack((vec_features, topic_vec))
final_set= hstack((temp, X))



from sklearn.preprocessing import normalize
final_set_normalized = normalize(final_set, norm='l1', axis=1)


# In[ ]:


scaler = MinMaxScaler()
features = scaler.fit_transform(topic_vec)
vec_features = sparse.csr_matrix(features)


# In[30]:


X=final_set
y=df['Rating']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=41)


# In[93]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
score


# In[94]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
score1= classifier.fit(X_train, y_train)
score


# In[ ]:


from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test)
accuracy = svm_model_linear.score(X_test, y_test)
accuracy


# In[25]:


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=50)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


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


# In[ ]:


data = [' '.join(text) for text in base]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)
temp= hstack((vec_features, X))

X=temp
y=df['Rating']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=41)

from sklearn.naive_bayes import MultinomialNB

mlt = MultinomialNB()
mlt.fit(X_train,y_train)
mlt.score(X_test, y_test)


# In[8]:


df.to_csv('backup.csv')


# In[12]:


df.head(10)

