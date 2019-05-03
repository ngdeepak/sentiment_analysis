

# Introduction:
	'''This task is a part of AI-hackathon conducted at Round Rock dell office to build a sentiment analysis model'''
 
# Objective:
	'''The objective is to build a multiclass predictive model which can classify the review in the following ratings:

	* negative
	* somewhat negative
	* neutral
	* somewhat positive
	* positive '''

#import required packages
#basics
import pandas as pd 
import numpy as np

#misc
import gc
import time
import warnings

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from PIL import Image
import matplotlib_venn as venn
#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
# import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
# Modeling
import lightgbm as lgb

# downloading the nltk corpus
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('punkt')

#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")
lem = WordNetLemmatizer()

#importing the dataset
train=pd.read_csv("train_1.csv")
test1=pd.read_csv("test_1.csv")
test2=pd.read_csv("test_1.csv")

# Shape of models
nrow_train=train.shape[0]
nrow_test1=test1.shape[0]
nrow_test2=test2.shape[0]
sum=nrow_train+nrow_test1+nrow_test2
print("       : train : test")
print("rows   :",nrow_train,":",nrow_test1,":",nrow_test2)
print("perc   :",round(nrow_train*100/sum),"   :",round(nrow_test1*100/sum),"   :",round(nrow_test2*100/sum))

# Replacing 'ReviewText' with 'Review'
train.rename(columns={'ReviewText':'Review'},inplace = True)    
actual_target = train['Rating']

# checking missing data proportion and replacing with "unknown"
print("Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)
print("Check for missing values in Test1 dataset")
null_check=test1.isnull().sum()
print(null_check)
print("filling NA with \"unknown\"")
print("Check for missing values in Test2 dataset")
null_check=test2.isnull().sum()
print(null_check)
print("filling NA with \"unknown\"")
train["Review"].fillna("unknown", inplace=True)
test1["Review"].fillna("unknown", inplace=True)
test2["Review"].fillna("unknown", inplace=True)

# plotting of distribution
x=train['Rating'].value_counts()
#plot
plt.figure(figsize=(15,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Rating-Type ', fontsize=12)
plt.show()


# Feature engineering:
# We've broadly classified feature engineering ideas into the following three groups
# Direct features:
# Features which are a directly due to words/content.We would be exploring the following techniques
# * Word frequency features
#     * Count features
#     * Bigrams
#     * Trigrams
# * Vector distance mapping of words (Eg: Word2Vec)
# * Sentiment scores
# 
# ## Indirect features:
# Some more experimental features.
# * count of sentences 
# * count of words
# * count of unique words
# * count of letters 
# * count of punctuations
# * count of uppercase words/letters
# * count of stop words
# * Avg length of each word

# merging all the reviews
merge=pd.concat([train.iloc[:,0:1],test1.iloc[:,0:1],test2.iloc[:,0:1]])
df=merge.reset_index(drop=True)

# creating meta features from review
#Sentense count in each comment:
    #  '\n' can be used to count the number of sentences in each comment
df['count_sent']=df["Review"].apply(lambda x: len(re.findall("\n",str(x)))+1)
# check for unique values of count of sentence
if df['count_sent'].nunique()==1:
    df.drop(['count_sent'],axis=1,inplace = True)
    
#Word count in each comment:
df['count_word']=df["Review"].apply(lambda x: len(str(x).split()))
#Unique word count
df['count_unique_word']=df["Review"].apply(lambda x: len(set(str(x).split())))
#Letter count
df['count_letters']=df["Review"].apply(lambda x: len(str(x)))
#punctuation count
df["count_punctuations"] =df["Review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#upper case words count
df["count_words_upper"] = df["Review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#title case words count
df["count_words_title"] = df["Review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
df["count_stopwords"] = df["Review"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
#Average length of the words
df["mean_word_len"] = df["Review"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

#derived features
#Word count percent in each comment:
df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
#derived features
#Punct percent in each comment:
df['punct_percent']=df['count_punctuations']*100/df['count_word']

#serperate train and tests features
train_feats=df.iloc[0:len(train),]
test_feats=df.iloc[len(train):,]
test1_feats=test_feats.iloc[0:len(test1),]
test2_feats=test_feats.iloc[len(test1):,]
del test_feats
#join the tags
train_tags=train['Rating']
train_feats=pd.concat([train_feats,train_tags],axis=1)

# corpus cleaning
corpus=merge.Review

# Aphost lookup dict
APPO = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 
        "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
        "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",
        "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", 
        "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
        "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", 
        "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",
        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
        "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 
        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
        "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", 
        "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
        "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 
        "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
        "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
        "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
        "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
        "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  
        "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 
        "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
        "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
        "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
        "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
        "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
        "you'll've": "you will have", "you're": "you are", "you've": "you have", "n't": "not", "'ve": "have"}

def clean(comment,purpose):
    
    comment=comment.lower()
    #remove \n
    comment=re.sub("\\n","",comment)
    
    #Split the sentences into words
    words=word_tokenize(comment)
    words=[APPO[word] if word in APPO else word for word in words]
#     table = str.maketrans('', '', string.punctuation)
#     words = [w.translate(table) for w in words]
    # remove remaining tokens that are not alphabetic
    words = [word for word in words if word.isalpha()]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    if purpose == "Topic Modeling":
        return words
    else:
        clean_sent=" ".join(words)
        return(clean_sent)

clean_corpus=corpus.apply(lambda x :clean(x,"TFIDF"))
# text_data = corpus.apply(lambda x:clean(x,"Topic Modeling")) # didn't try topic modeling as it was taking more time

# # Direct features:
# 
# ## 1)Count based features(for unigrams):
# Lets create some features based on frequency distribution of the words. Initially lets consider taking words one at a time (ie) Unigrams
# 
# Python's SKlearn provides 3 ways of creating count features.All three of them first create a vocabulary(dictionary) of words and then create a sparse matrix of word counts for the words in the sentence that are present in the dictionary. A brief description of them:
# * CountVectorizer
#     * Creates a matrix with frequency counts of each word in the text corpus
# * TF-IDF Vectorizer
#     * TF - Term Frequency -- Count of the words(Terms) in the text corpus (same of Count Vect)
#     * IDF - Inverse Document Frequency -- Penalizes words that are too frequent. We can think of this as regularization
# * HashingVectorizer
#     * Creates a hashmap(word to number mapping based on hashing technique) instead of a dictionary for vocabulary
#     * This enables it to be more scalable and faster for larger text coprus
#     * Can be parallelized across multiple threads
#         
# Using TF-IDF here.
# Note: Using the concatenated dataframe "merge" which contains both text from train and test dataset to ensure that the vocabulary that we create does not missout on the words that are unique to testset.

### Unigrams -- TF-IDF 
start_unigrams=time.time()
tfv = TfidfVectorizer(min_df=5,  max_features=10000, 
            strip_accents='unicode', analyzer='word',ngram_range=(1,1),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())

train_unigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
test1_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:train.shape[0]+test1.shape[0],])
test2_unigrams = tfv.transform(clean_corpus.iloc[train.shape[0]+test1.shape[0]:,])

# bigram features
tfv = TfidfVectorizer(min_df=5,  max_features=30000, 
            strip_accents='unicode', analyzer='word',ngram_range=(2,2),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_bigrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
test1_bigrams = tfv.transform(clean_corpus.iloc[train.shape[0]:train.shape[0]+test1.shape[0],])
test2_bigrams = tfv.transform(clean_corpus.iloc[train.shape[0]+test1.shape[0]:,])

# char grams
tfv = TfidfVectorizer(min_df=5,  max_features=30000, 
            strip_accents='unicode', analyzer='char',ngram_range=(1,4),
            use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())
train_charngrams =  tfv.transform(clean_corpus.iloc[:train.shape[0]])
test1_charngrams = tfv.transform(clean_corpus.iloc[train.shape[0]:train.shape[0]+test1.shape[0],])
test2_charngrams = tfv.transform(clean_corpus.iloc[train.shape[0]+test1.shape[0]:,])

# if 'count_sent' has more unique values more than 1, then include that 
SELECTED_COLS=['count_word', 'count_unique_word',
       'count_letters', 'count_punctuations', 'count_words_upper',
       'count_words_title', 'count_stopwords', 'mean_word_len',
       'word_unique_percent', 'punct_percent']
target_x = train_feats[SELECTED_COLS]
# TARGET_COLS=train_tags.columns
target_y=train_tags

# Lightgbm will use [0-5) for 5 classes, therefore changing the input
target1_y = target_y-1

#merging all features
from scipy.sparse import csr_matrix, hstack
train_x = hstack((train_bigrams,train_charngrams,train_unigrams,train_feats[SELECTED_COLS])).tocsr()
test1_x = hstack((test1_bigrams,test1_charngrams,test1_unigrams,test1_feats[SELECTED_COLS])).tocsr()
test2_x = hstack((test2_bigrams,test2_charngrams,test2_unigrams,test2_feats[SELECTED_COLS])).tocsr()

# # Lightgbm using train/validation split
# # predictions test1
param = {'num_leaves': 30,
         'min_data_in_leaf': 31, 
         'objective':'multiclass',
         'num_class':5,
         'max_depth': 10,
         'learning_rate': 0.1,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'multi_logloss',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 16,
         "random_state": 4590}
X_train, X_valid, y_train, y_valid = train_test_split(train_x, target1_y, test_size=0.20, random_state=41)
trn_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_valid, label=y_valid)
num_round = 10000
clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=50, early_stopping_rounds = 30)
oof = clf.predict(X_valid, num_iteration=clf.best_iteration)

# #     fold_importance_df = pd.DataFrame()
# #     fold_importance_df["Feature"] = df_train_columns
# #     fold_importance_df["importance"] = clf.feature_importance()
# #     fold_importance_df["fold"] = fold_ + 1
# #     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

predictions1 = clf.predict(test1_x, num_iteration=clf.best_iteration)
predictions2 = clf.predict(test2_x, num_iteration=clf.best_iteration)
print("logloss is {}".format(log_loss(y_valid,oof)))
OOF_df1 = pd.DataFrame(data=oof,columns=[0,1,2,3,4])
predicted_result1 = OOF_df1.idxmax(axis=1)
# accuracy on validation set
print("accuracy :{}".format(np.mean(predicted_result1.values==y_valid.values)))
test1_result = pd.DataFrame(data=predictions1,columns=[1,2,3,4,5])# mapping test back to original rating, as prediction will be in 0-4 for 1-5
test2_result = pd.DataFrame(data=predictions2,columns=[1,2,3,4,5])

# final prediction is just max probability column
test1['Prediction'] = test1_result.idxmax(axis=1)
test2['Prediction'] = test2_result.idxmax(axis=1)
test1.head()

# saving the final results
test1.to_csv("prediction_test1.csv",index = False)
test2.to_csv("prediction_test2.csv",index = False)

