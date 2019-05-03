import pandas as pd
file=pd.read_csv("train.csv")
 
file.head()
 
file.count()
 
#Grouping by the ratings
file.groupby('Rating').count()
 
#Cut the data into the same size for each rating
R1 = file.loc[file['Rating'] == 1][:50000]
R2 = file.loc[file['Rating'] == 2][:50000]
R3 = file.loc[file['Rating'] == 3][:50000]
R4 = file.loc[file['Rating'] == 4][:50000]
R5 = file.loc[file['Rating'] == 5][:50000]
 
#Appending each data set to an appended location
R1 = R1.append(R2).append(R3).append(R4).append(R5)
 
 
#Outputting the new training data set to csv
R1.to_csv("train_data_even_dist.csv", sep=',')