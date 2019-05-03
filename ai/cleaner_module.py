
# coding: utf-8

# In[78]:


#Imports

import re
import warnings
warnings.filterwarnings("ignore")
import pyodbc
import pandas as pd
import numpy as np
import spacy
import string
import pickle
import gensim
import scipy as sp
import sklearn
import sys
import nltk
np.random.seed(400)
pd.set_option('display.max_colwidth', -1)
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
from gensim import corpora
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nolemmatize=set(['windows','bios','cmos', 'media']) 
pd.set_option('display.max_colwidth', -1)
import dask.dataframe as dd
from dask.multiprocessing import get
from flashtext import KeywordProcessor
stemmer = SnowballStemmer("english")
stop_words = stopwords.words('english')
from gensim.corpora import Dictionary
from gensim.test.utils import get_tmpfile
from gensim.models import ldamodel
import gensim.corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import logging
logging.getLogger('').handlers = []
logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='clean.log',
                            filemode='w')


# In[ ]:


# main function
readData= pd.read_csv('train.csv')

# This is the main function which calls the respective stages

stage1= basicClean(readData)
stage2= generateMetaData(stage1)
stage3= negativeWordRemoval(stage2)
stage4= lemmatize_df(stage3)
processed_docs = []
for doc in stage4:
    processed_docs.append(preprocess(doc))
pickle_tp = open("data_tp.pickle","wb")
pickle.dump(processed_docs, pickle_tp)
stage3= stage3[['Rating','ReviewText','wc_f1','uwc_f2','wc_f3']]
stage3.to_csv('cleaned_df.csv')


# In[73]:


def basicClean(readData):
    
    # This function performs the basic cleaning (Removing duplicates from the data)
    
    logging.info("Entering stage 1")
    df= readData.copy()
    df.dropna(inplace=True)
    df= df[['ReviewText','Rating']]
    df['ReviewText']= df['ReviewText'].map(lambda x: re.sub(r'[^ a-zA-Z0-9]', ' ', str(x)))
    logging.info("Exiting stage 1")
    return df

def negativeWordRemoval(readData):
    
    # This function removes the negative words from the dataset
    
    logging.info("Entering stage 3")
    df= readData.copy()
    tags= [ 'wont','doesnt','wasnt','cant','isnt','unable','not','un','never','nothing','isnt','isn']
    for i in tags:
        df['ReviewText']= df['ReviewText'].map(lambda x: re.sub(r'\b' + i + r'\b', 'NegativeWord', x))
    df['ReviewText']= df['ReviewText'].map(lambda x:str(x))  
    logging.info("Exiting stage 3")
    return df                                        

def generateMetaData(readData):
    
    # This function generates the meta features 
    
    logging.info("Entering stage 2")
    df= readData.copy()
    df['tokens']= df['ReviewText'].map(lambda x: x.split()) # Generates the token split
    df['wc_f1']= df['tokens'].map(lambda x: len(x)) # This generates the word count
    df['uwc_f2']= df['tokens'].map(lambda x: len(set(x))) # This generates the unique words in the document
    df['wc_f3']= df['tokens'].map(lambda x: getWc(x)) # This generate
    logging.info("Exiting stage 2")
    return df
                                
def getWc(x):
    
    # This function generates words with more than 5 characters
    
    c=0
    for i in x:
        if(len(i)>=5):
            c=c+1       
    return c
                                                
def preprocess(text):
     # This function performs basic cleaning (converts text to lower case, removes the accent)
    result=[]
    for token in gensim.utils.simple_preprocess(text,deacc=True) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 1:
            result.append(token)
    return result

def removeLoggersAndUrl(x):
        # Removes the URLS
    x1= re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)
    x2= re.sub(r'^http?:\/\/.*[\r\n]*', '', x1, flags=re.MULTILINE)
    x3= x2.lower()
    x4= x3.strip()
    return x4

# This function is used to lemmatize the texts (multiprocessing)
def lemmatize_df(t):
    
    
    logging.info("Entering stage 4")
    result_lemma= t['ReviewText']
    output=[]
    docs=nlp.pipe(t['ReviewText'],batch_size=1000, n_threads=26)
    for doc in docs:
        lemmatized_text=[word.text if word.text in nolemmatize else word.lemma_ for word in doc]
        output.append(" ".join([word for word in lemmatized_text if word != '-PRON-']))
    logging.info("Exiting stage 4")
    return output


# In[54]:


stage3.to_csv('cleaned_df.csv')

