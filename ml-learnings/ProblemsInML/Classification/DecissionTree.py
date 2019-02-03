# Entropy 
# Information Gain (After splitting dataset reducing the Entrophy)
# Leaf Node (Carries the classification)
# Root Node (Before the split )
# Training dataset is the concrete answer which you have in hand

# Solution : Entrophy == 0

# Problem Statement
# Loan Repayment Prediction.

#%%
# impor the packages 
import numpy as np 
import pandas as pd
import os


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# load your data
#%%
data= pd.read_csv('data/breastCancer.csv')
data.columns

#%%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

#%%
data.diagnosis=[1 if each == "M" else 0 for each in data.diagnosis]
data.head()


#%%
data.info()

#%%
y=data.diagnosis.values

#normalization
x_data=data.drop(["diagnosis"], axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data) -np.min(x_data))

#%%
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)

#%%
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("score :",dt.score(x_test,y_test))
