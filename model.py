#!/usr/bin/env python
# coding: utf-8

# ## feature engineering

# In[1]:


# standard library imports
import csv
import datetime as dt
import json
import os
import statistics
import time

# third-party imports
import numpy as np
import pandas as pd
import requests

# google cloud storage
import gcsfs

import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv('steamspy_onehot.csv')

# number of languages
def cnt(var):
    return var.count(',')+1

# missing values considered as 1
res = []
for i in range(len(data.languages)):
    if pd.isna(data.languages[i]):
        tmp = 1
    else:
        tmp = cnt(data.languages[i])
    res.append(tmp)
    
data['n_lang'] = res

data['favorable_rate'] = data.positive/(data.positive+data.negative)

d = pd.DataFrame(data['developer'].value_counts().head(10)).reset_index()
p = pd.DataFrame(data['publisher'].value_counts().head(10)).reset_index()

data['top10_dev'] = [1 if i in d.index else 0 for i in data.developer]
data['top10_pub'] = [1 if i in p.index else 0 for i in data.publisher]

o = []
for i in data.owners:
    if i == '0 .. 20,000':
        j = 10000
    elif i == '20,000 .. 50,000':
        j = 35000
    elif i == '50,000 .. 100,000':
        j = 75000
    elif i == '100,000 .. 200,000':
        j = 150000
    elif i == '200,000 .. 500,000':
        j = 350000
    elif i == '500,000 .. 1,000,000':
        j = 750000
    elif i == '1,000,000 .. 2,000,000':
        j = 1500000
    elif i == '2,000,000 .. 5,000,000':
        j = 3500000
    elif i == '5,000,000 .. 10,000,000':
        j = 7500000
    elif i == '10,000,000 .. 20,000,000':
        j = 15000000
    elif i == '20,000,000 .. 50,000,000':
        j = 35000000
    elif i == '50,000,000 .. 100,000,000':
        j = 75000000
    else:
        j = 150000000
    o.append(j)
    
data['n_owner'] = o

data['if_dis'] = [1 if i != 0 else 0 for i in data.discount]


# ## lda

# ### preprocess

# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data['owners'][0].split(' .. ')


# In[6]:


data['owners_lbud'] = data['owners'].apply(lambda x: x.split(' .. ')[0])
data['owners_hbud'] = data['owners'].apply(lambda x: x.split(' .. ')[1])
int(data['owners_lbud'][0].replace(',', ''))


# In[7]:


data['owners_lbud'] = data['owners_lbud'].apply(lambda x: int(x.replace(',', '')))
data['owners_hbud'] = data['owners_hbud'].apply(lambda x: int(x.replace(',', '')))


# In[8]:


data['owners_mid'] = (data['owners_lbud'] + data['owners_hbud']) / 2
data.head()


# In[9]:


data_filtered = data.drop(['index', 'developer', 'publisher', 'score_rank', 'positive', 'negative',                        'userscore', 'owners', 'languages', 'genre', 'tags'], axis = 1)


# In[10]:


data_filtered[data_filtered.columns[pd.Series(data_filtered.columns).str.startswith('tag')]].sum().sort_values(ascending=False)[:10]


# In[11]:


data_filtered[data_filtered.columns[pd.Series(data_filtered.columns).str.startswith('genre')]].sum().sort_values(ascending=False)[:10]


# In[12]:


top_10tag = ["tag_Action", "tag_Indie", "tag_Adventure", "tag_Multiplayer", "tag_Singleplayer",              "tag_Strategy", "tag_Casual", "tag_Free to Play", "tag_RPG", "tag_Simulation"]

top_10genre = ["genre_Indie", "genre_Action", "genre_Casual", 'genre_Adventure', "genre_Strategy",               "genre_Simulation", "genre_RPG", "genre_EarlyAccess", 'genre_FreetoPlay', "genre_Sports"]


# In[13]:


data_filtered = data_filtered.dropna(axis=0)
data_filtered = data_filtered.reset_index()
data_filtered.shape


# In[14]:


data_tag_cluster = data_filtered.filter(regex='^tag',axis=1)
data_tag_cluster.shape
data_tag_cluster.shape


# In[15]:


data_tag_cluster['owners_mid'] = data_filtered['owners_mid']
data_tag_cluster['favorable_rate'] = data_filtered['favorable_rate']
data_tag_cluster.head()


# In[16]:


train_tagclus, test_tagclus = train_test_split(data_tag_cluster, test_size=0.2, random_state=42, shuffle=True)
train_tagclus


# In[17]:


X_cluster = train_tagclus.drop(['owners_mid', 'favorable_rate'], axis = 1)
X_cluster


# ### lda model

# In[18]:


pip install gensim


# In[19]:


from gensim.models import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary


# In[20]:


#generating dictionary and corpos from trainning dataset
tag_lda = []
for i in X_cluster.index:
    temp = []
    for col in X_cluster.columns:
        if col != 'index':
            for j in range(int(X_cluster[col][i])):
                temp.append(col)
    tag_lda.append(temp)
    
common_dictionary = Dictionary(tag_lda)
common_corpus = [common_dictionary.doc2bow(text) for text in tag_lda]


# In[21]:


# Set training parameters.
num_topics = 20
chunksize = 500 # size of the doc looked at every pass
passes = 20 # number of passes through documents
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = common_dictionary[0]  # This is only to "load" the dictionary.
id2word = common_dictionary.id2token

model = LdaModel(corpus=common_corpus, id2word=id2word, chunksize=chunksize,                    alpha='auto', eta='auto',                    iterations=iterations, num_topics=num_topics,                    passes=passes, eval_every=eval_every)


# In[22]:


get_ipython().system(' pip install pyLDAvis')


# In[23]:


import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[24]:


pyLDAvis.gensim_models.prepare(model, common_corpus, common_dictionary, sort_topics = False)


# In[25]:


#print out topic distribution
for i,topic in model.show_topics(formatted=True, num_topics=20, num_words=10):
    print(str(i)+": "+ topic)
    print()


# In[26]:


data_tag_cluster.head()


# In[27]:


data_tag_cluster.shape


# In[28]:


## preprocessing entire data
tag_lda_total = []
for i in data_tag_cluster.index:
    temp = []
    for col in data_tag_cluster.columns:
        if col != 'index' and col != 'owners_mid' and col != 'favorable_rate':
            for j in range(int(data_tag_cluster[col][i])):
                temp.append(col)
    tag_lda_total.append(temp)
    
common_dictionary_total = Dictionary(tag_lda_total)
common_corpus_total = [common_dictionary.doc2bow(text) for text in tag_lda_total]


# In[29]:


topic_decomposition = []
for i in range(len(common_corpus_total)):
    topic_decomposition.append(model[common_corpus_total[i]])


# In[30]:


data_tag_cluster['topic_decomposition'] = topic_decomposition


# In[31]:


data_filtered['topic_decomposition'] = topic_decomposition
data_filtered['major_topic'] = data_filtered['topic_decomposition'].apply(lambda x: sorted(x, key=lambda tup: tup[1])[-1][0])
data_filtered['major_topic_percentage'] = data_filtered['topic_decomposition'].apply(lambda x: sorted(x, key=lambda tup: tup[1])[-1][1])


# In[66]:


data_filtered.to_csv('gs://6893projectdata/steamspy_withlda.csv', index = False)


# ## sentiment analysis

# In[33]:


df = pd.read_csv('steamspy_data.csv')

import ast
def gettag(var):
    if var == '[]':
        return 'NA'
    else:
        var = ast.literal_eval(var)
        return list(var)[0]
df['tag'] = df.tags.apply(gettag)

df.tag.value_counts()


# 319 unqiue tags, making it impossible to conduct twitter analysis. 

# ## 1. popularity (dau)

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
d1 = data.drop(['index','appid','name','developer','publisher','score_rank','owners','initialprice','discount','languages',
                'genre','tags'], axis=1)
d1 = d1.dropna()
x = d1.drop('average_forever', axis=1)
y = d1.average_forever

# standard scaler
scaler = preprocessing.StandardScaler()
xs = scaler.fit_transform(x)
# pca to reduce dimensionality
pca = PCA(n_components=10)
xp = pca.fit_transform(x)
# both
xsp = pca.fit_transform(xs)

# train_test_split， default: testsize=25%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=1)
xtrains, xtests, ytrain, ytest = train_test_split(xs, y, random_state=1)
xtrainp, xtestp, ytrain, ytest = train_test_split(xp, y, random_state=1)
xtrainsp, xtestsp, ytrain, ytest = train_test_split(xsp, y, random_state=1)


# In[35]:


import numpy as np
# Symmetric mean absolute percentage error: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true)+np.abs(y_pred)+0.01)) # add 0.01 to avoid nan

def lr(x,x1,y,y1):
    model = LinearRegression()
    model.fit(x,y)
    ypred = model.predict(x1)
    smape = symmetric_mean_absolute_percentage_error(y1, ypred)
    print(f'smape = {smape}')
    
# baseline
lr(xtrain, xtest, ytrain, ytest)
# scaling
lr(xtrains, xtests, ytrain, ytest)
# pca
lr(xtrainp, xtestp, ytrain, ytest)
# scaling + pca
lr(xtrainsp, xtestsp, ytrain, ytest)


# linear regression with pca has the lowest smape of ~70%.

# In[36]:


from sklearn.ensemble import RandomForestRegressor

def rf(x,x1,y,y1):
    model = RandomForestRegressor()
    model.fit(x,y)
    ypred = model.predict(x1)
    smape = symmetric_mean_absolute_percentage_error(y1, ypred)
    print(f'smape = {smape}')
    
# baseline
rf(xtrain, xtest, ytrain, ytest)


# In[37]:


# scaling isn't vital for tree model, so test pca in rf
rf(xtrainp, xtestp, ytrain, ytest)


# baseline random forest has the lowest smape of ~4%. 

# In[38]:


# predict all games' dau using rf
model = RandomForestRegressor()
model.fit(xtrain,ytrain)
ypred_all = model.predict(x)


# ## 2. penetration (ownership)

# In[39]:


x = d1.drop('n_owner', axis=1)
y = d1.n_owner

# standard scaler
scaler = preprocessing.StandardScaler()
xs = scaler.fit_transform(x)
# pca to reduce dimensionality
pca = PCA(n_components=10)
xp = pca.fit_transform(x)
# both
xsp = pca.fit_transform(xs)

# train_test_split， default: testsize=25%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=1)
xtrains, xtests, ytrain, ytest = train_test_split(xs, y, random_state=1)
xtrainp, xtestp, ytrain, ytest = train_test_split(xp, y, random_state=1)
xtrainsp, xtestsp, ytrain, ytest = train_test_split(xsp, y, random_state=1)


# In[40]:


# baseline
lr(xtrain, xtest, ytrain, ytest)
# scaling
lr(xtrains, xtests, ytrain, ytest)
# pca
lr(xtrainp, xtestp, ytrain, ytest)
# scaling + pca
lr(xtrainsp, xtestsp, ytrain, ytest)


# In[41]:


rf(xtrain, xtest, ytrain, ytest)
rf(xtrainp, xtestp, ytrain, ytest)


# In[42]:


from sklearn.ensemble import GradientBoostingRegressor
def gbr(x,x1,y,y1):
    model = GradientBoostingRegressor()
    model.fit(x,y)
    ypred = model.predict(x1)
    smape = symmetric_mean_absolute_percentage_error(y1, ypred)
    print(f'smape = {smape}')
    
gbr(xtrain, xtest, ytrain, ytest)
gbr(xtrainp, xtestp, ytrain, ytest)


# random forest shot the best performance for predicting ownership. 

# In[43]:


model = RandomForestRegressor()
model.fit(xtrain,ytrain)
ypred_all_own = model.predict(x)


# ## 3. playability (ccu)

# In[44]:


x = d1.drop('ccu', axis=1)
y = d1.ccu

# standard scaler
scaler = preprocessing.StandardScaler()
xs = scaler.fit_transform(x)
# pca to reduce dimensionality
pca = PCA(n_components=10)
xp = pca.fit_transform(x)
# both
xsp = pca.fit_transform(xs)

# train_test_split， default: testsize=25%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=1)
xtrains, xtests, ytrain, ytest = train_test_split(xs, y, random_state=1)
xtrainp, xtestp, ytrain, ytest = train_test_split(xp, y, random_state=1)
xtrainsp, xtestsp, ytrain, ytest = train_test_split(xsp, y, random_state=1)


# In[45]:


# baseline
lr(xtrain, xtest, ytrain, ytest)
# scaling
lr(xtrains, xtests, ytrain, ytest)
# pca
lr(xtrainp, xtestp, ytrain, ytest)
# scaling + pca
lr(xtrainsp, xtestsp, ytrain, ytest)


# In[46]:


rf(xtrain, xtest, ytrain, ytest)
rf(xtrainp, xtestp, ytrain, ytest)


# In[47]:


gbr(xtrain, xtest, ytrain, ytest)
gbr(xtrainp, xtestp, ytrain, ytest)


# In[48]:


model = RandomForestRegressor()
model.fit(xtrain,ytrain)
ypred_all_ccu = model.predict(x)


# ## 4. feedback (favorable_rate)

# In[49]:


x = d1.drop('favorable_rate', axis=1)
y = d1.favorable_rate

# standard scaler
scaler = preprocessing.StandardScaler()
xs = scaler.fit_transform(x)
# pca to reduce dimensionality
pca = PCA(n_components=10)
xp = pca.fit_transform(x)
# both
xsp = pca.fit_transform(xs)

# train_test_split， default: testsize=25%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=1)
xtrains, xtests, ytrain, ytest = train_test_split(xs, y, random_state=1)
xtrainp, xtestp, ytrain, ytest = train_test_split(xp, y, random_state=1)
xtrainsp, xtestsp, ytrain, ytest = train_test_split(xsp, y, random_state=1)


# In[50]:


# baseline
lr(xtrain, xtest, ytrain, ytest)
# scaling
lr(xtrains, xtests, ytrain, ytest)
# pca
lr(xtrainp, xtestp, ytrain, ytest)
# scaling + pca
lr(xtrainsp, xtestsp, ytrain, ytest)


# In[51]:


rf(xtrain, xtest, ytrain, ytest)
rf(xtrainp, xtestp, ytrain, ytest)


# In[52]:


gbr(xtrain, xtest, ytrain, ytest)
gbr(xtrainp, xtestp, ytrain, ytest)


# In[53]:


model = RandomForestRegressor()
model.fit(xtrain,ytrain)
ypred_all_fr = model.predict(x)


# ## generate csv

# In[54]:


d1['i'] = d1.index
data['i'] = data.index
m = data[['appid', 'name', 'publisher','i']]
r = m.merge(d1, on='i', how='inner')


# In[55]:


l = ['appid', 'name', 'publisher'] + [col for col in d1.columns if 'genre' in col] 
target = r[l]
target['popularity'] = ypred_all
target['penetration'] = ypred_all_own
target['playability'] = ypred_all_ccu
target['feedback'] = ypred_all_fr


# In[56]:


def score_pop(var):
    return 25*((var-target.popularity.min())/(target.popularity.max()-target.popularity.min()))

def score_pen(var):
    return 25*((var-target.penetration.min())/(target.penetration.max()-target.penetration.min()))

def score_pla(var):
    return 25*((var-target.playability.min())/(target.playability.max()-target.playability.min()))

def score_fee(var):
    return 25*((var-target.feedback.min())/(target.feedback.max()-target.feedback.min()))

target['score'] = target.popularity.apply(score_pop) + target.penetration.apply(score_pen) + target.playability.apply(score_pla) + target.feedback.apply(score_fee)


# In[57]:


target.sort_values('score', ascending=False)


# In[64]:


target.to_csv('gs://6893projectdata/game_scorecard_lda.csv', index=False)


# ## generate csv for web interface

# In[59]:


def movecol(df, cols_to_move=[], ref_col='', place='After'):
    
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])


# In[60]:


web = target.drop(['popularity','penetration','playability','feedback'], axis=1)
web = movecol(web, 
             cols_to_move=['score'], 
             ref_col='publisher',
             place='After')


# In[61]:


web.iloc[:,2] = web.iloc[:,2].fillna('NA')


# In[65]:


web.to_csv('gs://6893projectdata/webdata.csv', index = False)


# In[ ]:




