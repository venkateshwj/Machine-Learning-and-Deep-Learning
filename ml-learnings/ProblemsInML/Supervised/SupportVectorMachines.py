# Support Vector Machines classifier.
# %%
# data analysis packages
import pandas as pd
import numpy as np

from sklearn import svm

#%%
# Visualize your data
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

recipes = pd.read_csv('data/recipes_muffins_cupcakes.csv')
print (recipes.head())

sns.lmplot('Flour', 'Sugar', data= recipes, hue='Type', palette='Set1', fit_reg=False,scatter_kws={"s":70} )

#%%
# preprocess our data 
type_label = np.where(recipes['Type'] == 'Muffin', 0 ,1 )
recipes_features = recipes.columns.values[1:].tolist()
recipes_features

ingredients = recipes[['Flour','Sugar']].values
print(ingredients)

#%%
# Fit model (Support Vector Classifier)
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)

#%%
# Get the separating Hyperlane
w= model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30,60)
yy = a * xx- (model.intercept_[0])/w[1]

#%% plot the parallels to separating hyperplane that passes throught the support vectors
b= model.support_vectors_[0]
yy_down = a* xx + (b[1] - a * b [0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

#%%
# lets plot dude.
sns.lmplot('Flour', 'Sugar', data= recipes, hue='Type', palette='Set1', fit_reg=False,scatter_kws={"s":70} )
plt.plot(xx, yy, linewidth =2 , color='black' )
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')


#%%
# Lets predict and check our mode by writing a simple function
def muffin_or_cupcake(flour, sugar):
    if (model.predict([[flour, sugar]]))==0:
         print (' It is a muffin')
    else:
        print (' It is a cupcake')

#%%
# Let check our model
muffin_or_cupcake(50,20)

#%%
# lets Check in graph to understand better.
sns.lmplot('Flour', 'Sugar', data= recipes, hue='Type', palette='Set1', fit_reg=False,scatter_kws={"s":70} )
plt.plot(xx, yy, linewidth =2 , color='black' )
plt.plot(50, 20, 'yo', markersize='15')

