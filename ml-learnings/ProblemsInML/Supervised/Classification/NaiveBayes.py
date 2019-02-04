#Formula 

#H -hypothesis
#E- Evidence

#P(H|E) = P(E|H) .P(H)
#         __________
#            P( E)

#Advantages : Less training data, Handles both continuous and discrete data, Fast in real time predictions.

# Problem: Lets check a news article  and classify


#%%
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups

# Visualize your data
#%%
data = fetch_20newsgroups()
data.target_names

#%%
# Define the categories
categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

# Train data on these categories
train = fetch_20newsgroups(subset='train', categories=categories)

# Test data on these categories
test = fetch_20newsgroups(subset='test', categories=categories)

# priniting training data article 5
print (train.data[5])

# priniting test data article 5
print (test.data[5])


#%%
# Import packages for NB
# to remove duplicate words
from sklearn.feature_extraction.text import TfidfVectorizer

# to import Naive bayes AG
from sklearn.naive_bayes import MultinomialNB

# Creating a pipe line for naive bayes model
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(),MultinomialNB())

#%%
# training the model with train data.
print (train.target)
model.fit(train.data, train.target)

#%%
# Label test data

labels = model.predict(test.data)

#%%
# Confusion matrix and heat map
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square= True, annot = True, fmt='d',
cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)

# plot the heat map
plt.xlabel('true label')
plt.ylabel('predicted label')

#Let's check our data
#%%
def predict_category(s, train=train, model=model):
    pred= model.predict([s])
    return train.target_names[pred[0]]

#%%
# predict_category('rolls royce')
# predict_category('Space shuttle')
predict_category('Venkatesh waralu jayachandran')





