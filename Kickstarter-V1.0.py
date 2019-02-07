#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import string

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings
from collections import Counter


# In[9]:


# # visualization

# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline

# # import plotly.tools as tls
# import plotly.offline as py
# from plotly.offline import init_notebook_mode, iplot, plot
# import plotly.graph_objs as go
# init_notebook_mode(connected=True)
# import warnings
# from collections import Counter


# In[6]:


# Libraries
import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split


# In[7]:


df = pd.read_csv('ks-projects-201801.csv',encoding ='latin1')


# In[8]:


df.head(5)


# In[9]:


df.describe()


# In[10]:


print(df.shape)
print(df.info())


# In[11]:


percentual_sucess = round(df["state"].value_counts() / len(df["state"]) * 100,2)

print("State Percentual in %: ")
print(percentual_sucess)

state = round(df["state"].value_counts() / len(df["state"]) * 100,2)

labels = list(state.index)
values = list(state.values)

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))

layout = go.Layout(title='Distribuition of States', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)


# In[ ]:


percentual_sucess = round(df["state"].value_counts() / len(df["state"]) * 100,2)

print("State Percentual in %: ")
print(percentual_sucess)

state = round(df["state"].value_counts() / len(df["state"]) * 100,2)

labels = list(state.index)
values = list(state.values)

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))

layout = go.Layout(title='Distribuition of States', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)


# In[ ]:


df_failed = df[df["state"] == "failed"]
df_sucess = df[df["state"] == "successful"]

#First plot
trace0 = go.Histogram(
    x= np.log(df.usd_goal_real + 1).head(100000),
    histnorm='probability', showlegend=False,
    xbins=dict(
        start=-5.0,
        end=19.0,
        size=1),
    autobiny=True)

#Second plot
trace1 = go.Histogram(
    x = np.log(df.usd_pledged_real + 1).head(100000),
    histnorm='probability', showlegend=False,
    xbins=dict(
        start=-1.0,
        end=17.0,
        size=1))

# Add histogram data
x1 = np.log(df_failed['usd_goal_real']+1).head(100000)
x2 = np.log(df_sucess["usd_goal_real"]+1).head(100000)

trace3 = go.Histogram(
    x=x1,
    opacity=0.60, nbinsx=30, name='Goals Failed', histnorm='probability'
)
trace4 = go.Histogram(
    x=x2,
    opacity=0.60, nbinsx=30, name='Goals Sucessful', histnorm='probability'
)


data = [trace0, trace1, trace3, trace4]
layout = go.Layout(barmode='overlay')

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[ [{'colspan': 2}, None], [{}, {}]],
                          subplot_titles=('Failed and Sucessful Projects',
                                          'Goal','Pledged'))

#setting the figs
fig.append_trace(trace0, 2, 1)
fig.append_trace(trace1, 2, 2)
fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 1)

fig['layout'].update(title="Distribuitions",
                     height=500, width=900, barmode='overlay')
iplot(fig)


# In[ ]:


main_cats = df["main_category"].value_counts()
main_cats_failed = df[df["state"] == "failed"]["main_category"].value_counts()
main_cats_sucess = df[df["state"] == "successful"]["main_category"].value_counts()


# In[ ]:


sucess = df[df["state"] == "successful"]["main_category"].value_counts()
#First plot
trace0 = go.Bar(
    x=main_cats_failed.index,
    y=main_cats_failed.values,
    name="Failed Category's"
)
#Second plot
trace1 = go.Bar(
    x=main_cats_sucess.index,
    y=main_cats_sucess.values,
    name="Sucess Category's"
)
#Third plot
trace2 = go.Bar(
    x=main_cats.index,
    y=main_cats.values,
    name="All Category's Distribuition"
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('Failed','Sucessful', "General Category's"))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title="Main Category's Distribuition",bargap=0.05)
iplot(fig)


# In[ ]:


df['main_category'].value_counts().plot.bar()
plt.show()


# In[ ]:


df['currency'].value_counts().plot.bar()
plt.show()

df['country'].value_counts().plot.bar()
plt.show()


# In[ ]:


df['state'].value_counts().plot.bar()
plt.show()


# In[ ]:


df = df[(df['state'] == 'failed') | (df['state'] == 'successful')].copy()
print(df.shape)


# In[ ]:


# Delete column => 
# 'ID', 'name', 'category', 'usd pledged', 'usd_pledged_real'

df = df.drop('ID', 1)
    
df = df.drop('name', 1)

#df = df.drop('category', 1)

df = df.drop('usd pledged', 1)
    
df = df.drop('usd_pledged_real', 1)

df = df.drop('backers', 1)

print(df.shape)


# In[ ]:


# Create new column
# 'duration_days' = 'deadline' - 'launched'

df['launched'] = pd.to_datetime(df['launched'])
df['deadline'] = pd.to_datetime(df['deadline'])

df['duration_days'] = df['deadline'].subtract(df['launched'])
df['duration_days'] = df['duration_days'].astype('timedelta64[D]')


# In[ ]:


df = df.drop('launched', 1)

df = df.drop('deadline', 1)

df = df.drop('pledged', 1)


# In[ ]:


df = df[(df['goal'] <= 100000) & (df['goal'] >= 1000)].copy()
df.shape


# In[ ]:


# Encoding column 'state',
# failed = 0, successful = 1

df['state'] = df['state'].map({
        'failed': 0,
        'successful': 1         
})


# In[ ]:


print(df.shape)
df.head(5)


# In[ ]:


# We use one-hot-codding

df = pd.get_dummies(df, columns = ['category'])


# In[ ]:


# We use one-hot-codding

df = pd.get_dummies(df, columns = ['main_category'])


# In[ ]:


# Rename 'main_category_Film & Video' to 'main_category_Film'

df.rename(columns={"main_category_Film & Video": "main_category_Film"}, inplace=True)
print('DONE')


# In[ ]:



print(df.columns)
print(df.shape)


# In[ ]:


# We use one-hot-codding

df = pd.get_dummies(df, columns = ['currency'])


# In[ ]:


print(df.columns)
print(df.shape)


# In[ ]:


# use one-hot-coddsing

df = pd.get_dummies(df, columns=['country'])


# In[ ]:


print(df.columns)
print(df.shape)


# In[ ]:


# Upload data
name = pd.read_csv('ks-projects-201801.csv',encoding ='latin1')


# In[ ]:


# We use only 'name' & 'state'
name = name.drop(['ID', 'category', 'main_category', 'currency', 'deadline',
       'goal', 'launched', 'pledged', 'backers', 'country',
       'usd pledged', 'usd_pledged_real', 'usd_goal_real'], 1)
print(name.shape)

name = name[(name['state'] == 'failed') | (name['state'] == 'successful')].copy()
print(name.shape)

# Encoding column 'state',
# failed = 0, successful = 1
name['state'] = name['state'].map({
        'failed': 0,
        'successful': 1         
})


# In[ ]:


# column 'name' to string
name['name'] = name['name'].astype(str)


# In[ ]:


# split each "name"
name['name'] = name['name'].str.split()
name.head()


# In[ ]:


# failed, successful, canceled, undefined, live, suspended
# check key word

i = 0
for n in name['name']:
    if 'successful' in n:
        i = i+1
    if 'failed' in n:
        i = i+1
        
print(i)

# it`s good. We dont need clean key word


# In[ ]:


# clean each name. We need 'name' without punctuation

name['name'] = name['name'].apply(lambda x:' '.join([i for i in x if i not in string.punctuation]))


# In[ ]:


# all words have small letters

name['name'] = name['name'].str.lower()


# In[ ]:


# Filter out Stop Words
# Import stopwords with nltk.

from nltk.corpus import stopwords
stop = stopwords.words('english')

name['name'] = name['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[ ]:


# to string

name['name'] = name['name'].str.split()
name.head()


# In[ ]:


narch, Espresso, Bar]	1
# failed, successful, canceled, undefined, live, suspended
# check key word

i = 0
for n in name['name']:
    if 'successful' in n:
        i = i+1
    if 'failed' in n:
        i = i+1
        
print(i)

# it`s good. We dont need clean key word


# In[ ]:


# clean each name. We need 'name' without punctuation

name['name'] = name['name'].apply(lambda x:' '.join([i for i in x if i not in string.punctuation]))


# In[ ]:


# all words have small letters

name['name'] = name['name'].str.lower()


# In[ ]:


# Filter out Stop Words
# Import stopwords with nltk.

from nltk.corpus import stopwords
stop = stopwords.words('english')

name['name'] = name['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[ ]:


# to string

name['name'] = name['name'].str.split()
name.head()


# In[ ]:


# Stem Words
# Stemming refers to the process of reducing each word to its root or base.

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

name['name'] = name['name'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[ ]:


name.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


# In[ ]:


bag_of_words = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(name['name'])


# In[ ]:


print(df.shape)
df.head()


# In[ ]:


y = df['state']

print(y.shape)
y.head(5)


# In[ ]:


df = df.drop('state', 1)


# In[ ]:


# Split dataframe into random train and test subsets

X_train, X_test, Y_train, Y_test = train_test_split(
    df,
    y, 
    test_size = 0.1,
    random_state=42
)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[ ]:


# Logistic Regression
# 60.86

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

acc_log = round(logreg.score(X_test, Y_test) * 100, 2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


#68.57

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

acc_knn = round(knn.score(X_test, Y_test) * 100, 2)
acc_knn


# In[ ]:


# Linear SVC
#62.01

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)
acc_linear_svc


# In[ ]:


# Decision Tree
#77.86

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest
# 77.86

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
acc_random_forest


# In[ ]:


bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X_train, Y_train)

acc_bdt = round(bdt.score(X_test, Y_test) * 100, 2)
acc_bdt


# In[ ]:


clf_gb = GradientBoostingClassifier(n_estimators=100, 
                                 max_depth=1, 
                                 random_state=0)
clf_gb.fit(X_train, Y_train)

acc_clf_gb = round(clf_gb.score(X_test, Y_test) * 100, 2)
acc_clf_gb


# In[ ]:


mlp = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5, 
                    hidden_layer_sizes=(21, 2), 
                    random_state=1)

mlp.fit(X_train, Y_train)

acc_mlp = round(mlp.score(X_test, Y_test) * 100, 2)
acc_mlp


# In[ ]:


bagging = BaggingClassifier(
    KNeighborsClassifier(
        n_neighbors=8,
        weights='distance'
        ),
    oob_score=True,
    max_samples=0.5,
    max_features=1.0
    )
clf_bag = bagging.fit(X_train,Y_train)

acc_clf_bag = round(clf_bag.score(X_test, Y_test) * 100, 2)
acc_clf_bag


# In[ ]:


clf_lgbm = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        colsample_bytree=.8,
        subsample=.8,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01
    )

clf_lgbm.fit(X_train, 
        Y_train,
        eval_set= [(X_train, Y_train), (X_test, Y_test)], 
        eval_metric='auc', 
        verbose=0, 
        early_stopping_rounds=30
       )

acc_clf_lgbm = round(clf_lgbm.score(X_test, Y_test) * 100, 2)
acc_clf_lgbm


# In[ ]:


models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest',   
              'Linear SVC', 
              'Decision Tree', 'BaggingClassifier',
             'AdaBoostClassifier', 'GradientBoostingClassifier',
             'LGBMClassifier'],
    'Score': [acc_knn, acc_log, 
              acc_random_forest,   
              acc_linear_svc, acc_decision_tree,
             acc_clf_bag, acc_bdt, acc_clf_gb, 
              acc_clf_lgbm]})
models.sort_values(by='Score', ascending=False)

