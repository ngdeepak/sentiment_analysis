#Importing the necessary libraries
import fasttext as ft
import pandas as pd
import numpy as np
import re
 
 
 
 
#Using fasttext model to train data using supervised learning model and the even distrubted data
classifier = ft.supervised('train_data_even_dist.txt', 'model', label_prefix='__label__')
 
 
#Reading in the test file Generic Reviews to test the data
df = pd.read_csv('test1_generic_reviews.csv')
 
#Cleaning the data to replace the NAN values
df.fillna('',inplace=True)
#Converting each element in df into a list with 1 element in order for fasttext to predict
df_list_review = df['ReviewText'].apply(lambda x:[x])
 
 
#Creating empty list to store all the predicted ratings
ratings=[]
 
#Looping through the list of reviews and extracting each element.
#Using fast text predict method with each element of review
for i in df_list_review.tolist():
   
  if i[0] != '':
    labels = classifier.predict(i)
    labels = re.findall(r'\d+', str(labels[0]))
    ratings.append([i[0],labels[0]])
  else :
    labels = 0
    ratings.append([i[0],labels])
print (ratings)
 
#Outputting the ratings into a dataframe
df_output=pd.DataFrame(ratings)
 
#outputting the ratings to a cvs
df_output.to_csv("test2output_generic_evendist.csv")
 
 
   
   
 
#Running prediction for dell reviews
#Reading in the test file Generic
df = pd.read_csv('testB_dell_reviews.csv')
df.fillna('',inplace=True)
df_list_review = df['ReviewText'].apply(lambda x:[x])
 
 
#Looping through the list of reviews and extracting each element.
#Using fast text predict method with each element of review
print ()
ratings=[]
count=0
for i in df_list_review.tolist():
  if i[0] != '':
    labels = classifier.predict(i)
    labels = re.findall(r'\d+', str(labels[0]))
    ratings.append([i[0],labels[0]])
  else :
    labels = 0
    ratings.append([i[0],labels])
 
   
  
 
 
#Outputting the ratings into a dataframe and the csv
df_output=pd.DataFrame(ratings)
df_output.to_csv("test2_output_dell_evendist.csv")