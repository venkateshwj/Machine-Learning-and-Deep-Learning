# Entropy
# Information Gain (After splitting dataset reducing the Entrophy)
# Leaf Node (Carries the classification)
# Root Node (Before the split )
# Training dataset is the concrete answer which you have in hand

# Goal : Entrophy == 0

# Problem Statement
# Loan Repayment Prediction.

#%%
# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


#%%
#Import Dataset
balance_data = pd.read_csv('data/Decision_Tree_ Dataset.csv', sep= ',', header= 0)
#1. Intial Payment, 2. Last Payment, 3. Credit Score, 4. house Number
#5. Result.
balance_data.head()

#%%
# Check the rows and columns
print ("Dataset Lenght:: "), len(balance_data)
print ("Dataset Shape:: "), balance_data.shape


#%%
# Data in ( All the below data gave the 5th column)
# so making X question.
#1. Intial Payment, 2. Last Payment, 3. Credit Score, 4. house Number
X= balance_data.values[:, 0:5]
print(X)

# This is the answer in the data set
# Answer for the X data 
#%%%
Y= balance_data.values[:, 5:6]
print(Y)

#%%
# Split the datat
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#%%
print ("X train:: ")
print(X_train)

print ("X test:: ")
print(X_test)

print ("Y Train:: ")
print(y_train)


print ("Y test:: ")
print(y_test)

#%%
# Thress Splits and alteast 5 Leaf after split.
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

#%%
# X_test can be the live data to arrive the result
y_pred_en = clf_entropy.predict(X_test)
print(y_pred_en)

#%%
print ("Accuracy is "), accuracy_score(y_test,y_pred_en)*100


# Breast cancer statement
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
data.head()

#%%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.head()

#%%
data.diagnosis=[1 if each == "M" else 0 for each in data.diagnosis]
data.head()


#%%
data.info()

#%%
y=data.diagnosis.values
print(y)

#%%
#normalization
x_data=data.drop(["diagnosis"], axis=1)
print(x_data)

#%%
x=(x_data-np.min(x_data))/(np.max(x_data) -np.min(x_data))
print(x)

#%%
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)
print(x_test)
print(x_train)
print(y_train)
print(y_test)


#%%
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("score :",dt.score(x_test,y_test))
