# Note: This is a python 2.7


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

### Imputed Data

train_impute = pd.read_csv("data/impute-user-train.csv",index_col=0)
X_test_impute = pd.read_csv("data/impute-user-test.csv",index_col=0)

test_idx = X_test_impute.index
y_train_impute = train_impute['country_destination'].values
X_train_impute = train_impute.drop(labels='country_destination',axis=1).values


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X=X_train_impute,y=y_train_impute)


probs = rf.predict_proba(X=X_test_impute)
y_hat_rf = classify(probs)

submission = getSubmissionFile(user_idx=test_idx,predictions=y_hat_rf,k=5,country_map=country_label_df)
submission.to_csv("output/predictions/rf_raw_impute.csv")








