# Why Random Forrest? 
# NO Overfitting, High Accuracy, estimates missing data.
#
# Nothing but multiple decission trees
#
#USE THIS WHEN THERE IS MISSING DATA
#
#
#
#

# Entropy
# Information Gain (After splitting dataset reducing the Entrophy)
# Root Node (Before the split )
# Leaf Node (Carries the classification)
# DECISSION NODE.
# Training dataset is the concrete answer which you have in hand

# Problem : Species of FLower.

#%%
# Import Libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#Randomness is zero for practice.
np.random.seed(0)

#%%
# Load the Iris dataset
iris = load_iris()
print(iris.feature_names)
df = pd.DataFrame(iris.data, columns = iris.feature_names)

#%%
print (iris)

#%%
df.head()


#%%
# Adding new column for the speices name.
print(iris.target_names)
print(iris.target)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

#%%
# Adding one more column , Creating train and test data.
# train the 75 % of data so 75 % will have true values
# 25 % will have false
df['is_train'] = np.random.uniform(0,1, len(df)) <= .75
df.head()


#%%
# now lets have the train and test data.
# is_train == true goes to train data and the oposite as test data
train, test = df[df['is_train'] == True ] , df[df['is_train'] == False ]

#%%
# Lets see the total lenght of df, train and test
print('Total data length:', len(df))
print('Train data length:', len(train))
print('Test data length:', len(test))

#%%
# let see the column names again
features = df.columns[0:4]
features

#%%
# convert 'setosa' to digits
y = pd.factorize(train['species'])[0]
y

#%%
# now the RANDOM FORREST CLASSIFIER
# n_jobs for local computer just for prioritization
clf = RandomForestClassifier(n_jobs=2, random_state=0)
# Question(actual data) is train[features] and y is answers 
# we are fitting it 
clf.fit(train[features], y)

#%%
test[features]

#%%
# Lets test our model
print(features)

clf.predict(test[features])

#%%
# lets predict and see the result for 3 leafs and 20 nodes
# The maximum combination is the result.
clf.predict_proba(test[features])[10:20]

# 13 rows says setose and 14th one says either 2 nd or 3 rd leaf

#%%
# For human readable format lets convert the digits to speices name
preds = iris.target_names[clf.predict(test[features])]
preds[0:5]

#%%
test['species'].head()

#%%
preds

#%%
print(len(test['species']))
print(len(preds))

#%%
print(test['species'])

#%%
print(preds[::])


#%%
# Confusion matrix 
# Lets compare our actual data and predicted data
pd.crosstab(test['species'], preds, rownames=['Actual Species'],
 colnames=['Predicted Species'])

 # Model accuracy is (13+5+12)= 30/32 which 93 %

#%%
# Lets tet the model by giving input.
preds = iris.target_names[clf.predict([[5.0,3.6,1.4,2.0],[5.0,3.6,1.4,2.0]])]
preds

 
