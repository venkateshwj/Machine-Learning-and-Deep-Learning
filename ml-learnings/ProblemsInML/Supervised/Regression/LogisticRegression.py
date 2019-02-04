# Liner regression vs logistic regression
# Scenario : When venkat studied for x hours his marks also gets increase based on the number of hours.
# for the above scenario we can use linear regression to find it. 
# but, how is it possible to find venkat passed or not. The answer is SIGMOID

# Problem statement: Classify Tumor is 'Malignant' or 'benign'
#%%
# import libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


#%%
# load data set
data = pd.read_csv('data/tumor.csv')
data.head()

#%%
# the columns which are intrested in 
sns.jointplot('radius_mean', 'texture_mean', data=data)

#%%
sns.heatmap(data.corr())

#%%
data.isnull().sum()

#%%
data.columns
X= data[['radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst']]
y = data['diagnosis']
X.head()
y.head()

#%%
from sklearn.model_selection import train_test_split
# 30 percent to test it 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)

#%%
# Log model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#%%
y_pred = logmodel.predict(X_test)
print(y_pred)

#%%
# How good the model is ? lets check
from sklearn.metrics import classification_report
print (classification_report(y_test, y_pred))
