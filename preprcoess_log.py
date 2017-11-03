
# coding: utf-8

# Creating new features from the sessions data.

# Read in needed packages.

# In[1]:

import csv
import os
import numpy
import pandas


# Read in data.

# In[2]:

#set working directory
os.chdir('/Users/brinberg/Desktop/ist-557-final/data')


# In[3]:

#reading in data
sessions = pandas.read_csv('sessions.csv', encoding='utf-8')
sessions.head(n = 100)


# In[4]:

sessions.shape


# In[5]:

#descriptives for action, i.e., counting the total number of different actions
total_actions = sessions['action'].value_counts()
print (total_actions)


# Descriptives by user.

# In[6]:

row_count = sessions.groupby('user_id').count()
print (row_count)


# In[7]:

#total seconds elapsed per user
total_sec = sessions.groupby('user_id')[['secs_elapsed']].sum()
print (total_sec)


# In[8]:

#creating data set of new features that I will continually add to
new_features = pandas.concat([row_count, total_sec], axis=1)
new_features.head()


# In[9]:

#average seconds elapsed per user
avg_sec = sessions.groupby('user_id')[['secs_elapsed']].mean()
print (avg_sec)


# In[10]:

#adding feature to new_features
new_features = pandas.concat([new_features, avg_sec], axis=1)
new_features.head()


# In[11]:

#rename columns so there isn't any overlap
new_features.columns.values[5] = 'total_secs'
new_features.columns.values[6] = 'avg_secs'
new_features.head()


# In[12]:

#frequency of action by user

action_freq = pandas.DataFrame({'action_count' : sessions.groupby( ["user_id", "action"] ).size()}).reset_index()
print (action_freq)


# In[13]:

type(action_freq)
list(action_freq)


# In[14]:

#change action_freq from long to wide
action_wide = action_freq.pivot(index = 'user_id', columns = 'action', values = 'action_count')
print (action_wide)


# In[15]:

#add action type info to new_features data set
new_features = pandas.concat([new_features, action_wide], axis=1)
new_features.head()


# In[16]:

#frequency of action detail by user

action_detail_freq = pandas.DataFrame({'action_detail_count' : sessions.groupby( ["user_id", "action_detail"] ).size()}).reset_index()
print (action_detail_freq)


# In[17]:

#change action_detail_freq from long to wide
action_detail_wide = action_detail_freq.pivot(index = 'user_id', columns = 'action_detail', values = 'action_detail_count')
print (action_detail_wide)


# In[18]:

#add action type info to new_features data set
new_features = pandas.concat([new_features, action_detail_wide], axis=1)
new_features.head()


# In[19]:

#examine names of new_features columns
list(new_features)


# In[22]:

#fill all NAs in new_features with zeros (since all features are frequencies)
new_features1 = new_features.fillna(0)
print (new_features1)


# In[23]:

#output new_features to csv
new_features1.to_csv("sessions_new_features.csv")

