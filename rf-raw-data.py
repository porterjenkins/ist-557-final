import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from predictionFunctions import *

## Inverse Transform Label Encoder

countries = ['AU','CA','DE','ES','FR','GB','IT','NDF','NL','PT','US','other']
country_labels = dict(zip(range(len(countries)),countries))

country_label_df = DataFrame.from_dict(country_labels,orient='index')
country_label_df.columns = ['country']

### Raw data

train = pd.read_csv("data/raw-user-train.csv",index_col=0)
train.dropna(inplace=True)
X_test = pd.read_csv("data/raw-user-test.csv",index_col=0)
X_test.dropna(inplace=True)


test_idx = X_test.index
y_train = train['country_destination'].values
X_train = train.drop(labels='country_destination',axis=1).values


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X=X_train,y=y_train)

probs = rf.predict_proba(X=X_test)
y_hat_rf = classify(probs)

submission = getSubmissionFile(user_idx=test_idx,predictions=y_hat_rf,k=5,country_map=country_label_df)
submission.to_csv("output/predictions/rf_raw_data_drop_na.csv")









