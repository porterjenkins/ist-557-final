
# coding: utf-8

# Read in needed libraries.

# In[28]:

import csv
import os
import numpy
import pandas

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt


# Read in data set.



# In[3]:

#reading in data
train = pandas.read_csv('data/train_users_2.csv', encoding='utf-8')
train.head(n = 100)


# In[4]:

train.shape


# In[5]:

train.hist(figsize=(16,12))
plt.savefig("output/plots/train-hist.png")
plt.clf()
plt.close()

# In[6]:

#the age variable looks a bit off (i.e., some people entered the year they were born instead of their age), 
#so I'm going to follow some pre-processing steps from 
#https://github.com/davidgasquez/kaggle-airbnb/blob/master/notebooks/Preprocessing.ipynb

user_with_year_age_mask = train['age'] > 1000
train.loc[user_with_year_age_mask, 'age'] = 2015 - train.loc[user_with_year_age_mask, 'age']


# In[7]:

train.loc[(train['age'] > 100) | (train['age'] < 18), 'age'] = -1


# In[9]:

#modifying date information
#again, using code from https://github.com/davidgasquez/kaggle-airbnb/blob/master/notebooks/Preprocessing.ipynb
train['date_account_created'] = pandas.to_datetime(train['date_account_created'], errors='ignore')
train['date_first_active'] = pandas.to_datetime(train['timestamp_first_active'], format='%Y%m%d%H%M%S')


# In[11]:

#Convert to DatetimeIndex:
date_account_created = pandas.DatetimeIndex(train['date_account_created'])
date_first_active = pandas.DatetimeIndex(train['date_first_active'])


# In[12]:

#split dates into day, week, month, year:
train['day_account_created'] = date_account_created.day
train['weekday_account_created'] = date_account_created.weekday
train['week_account_created'] = date_account_created.week
train['month_account_created'] = date_account_created.month
train['year_account_created'] = date_account_created.year
train['day_first_active'] = date_first_active.day
train['weekday_first_active'] = date_first_active.weekday
train['week_first_active'] = date_first_active.week
train['month_first_active'] = date_first_active.month
train['year_first_active'] = date_first_active.year


# In[13]:

#Get the difference(time lag) between the date in which the account was created and when it was first active:
train['time_lag'] = (date_account_created.values - date_first_active.values).astype(int)


# In[14]:

#drop duplicated columns
drop_list = [
    'date_account_created',
    'date_first_active',
    'timestamp_first_active'
]

train.drop(drop_list, axis=1, inplace=True)


# In[15]:

train.head(n = 10)


# In[16]:

list(train)


# In[45]:

train.hist(figsize=(16,12))
plt.savefig("output/plots/train-hist-clean.png")
plt.clf()
plt.close()


# In[19]:

#making sure categorical variables are categorical
train["gender"] = train["gender"].astype('category')
train["signup_method"] = train["signup_method"].astype('category')
train["language"] = train["language"].astype('category')
train["affiliate_channel"] = train["affiliate_channel"].astype('category')
train["affiliate_provider"] = train["affiliate_provider"].astype('category')
train["first_affiliate_tracked"] = train["first_affiliate_tracked"].astype('category')
train["signup_app"] = train["signup_app"].astype('category')
train["first_device_type"] = train["first_device_type"].astype('category')
train["first_browser"] = train["first_browser"].astype('category')
train["country_destination"] = train["country_destination"].astype('category')


# In[20]:

#count descriptives 
train.describe()


# In[24]:

print(train["gender"].describe())
print(train["signup_method"].describe())
print(train["language"].describe())
print(train["affiliate_channel"].describe())
print(train["affiliate_provider"].describe())
print(train["first_affiliate_tracked"].describe())
print(train["signup_app"].describe())
print(train["first_device_type"].describe())
print(train["first_browser"].describe())
print(train["country_destination"].describe())


# In[35]:

#plots for categorical variables
#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['gender'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/gender-freq.png")
plt.clf()
plt.close()


# In[36]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['signup_method'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/signup-freq.png")
plt.clf()
plt.close()


# In[37]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['language'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/language-freq.png")
plt.clf()
plt.close()


# In[38]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['affiliate_channel'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/channel-freq.png")
plt.clf()
plt.close()

# In[39]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['affiliate_provider'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/provider-freq.png")
plt.clf()
plt.close()

# In[40]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['first_affiliate_tracked'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/first-affiliate-freq.png")
plt.clf()
plt.close()


# In[41]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['signup_app'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/signup-app-freq.png")
plt.clf()
plt.close()



# In[42]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['first_device_type'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/device-freq.png")
plt.clf()
plt.close()



# In[43]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['first_browser'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/browser-freq.png")
plt.clf()
plt.close()



# In[44]:

#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

train['country_destination'].value_counts().plot(kind='bar',figsize=(16,12))
plt.savefig("output/plots/destination-freq.png")
plt.clf()
plt.close()



# In[31]:

#scatterplot matrix
from pandas.plotting import scatter_matrix

my_scatter = scatter_matrix(train, alpha=0.2, figsize=(25, 25), diagonal='kde')

plt.savefig("output/plots/scatter-matrix.png")
plt.clf()
plt.close()



# In[ ]:



