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

train_impute = pd.read_csv("data/raw-user-train.csv",index_col=0)
train_impute.dropna(inplace=True)
X_test_impute = pd.read_csv("data/raw-user-test.csv",index_col=0)

test_idx = X_test_impute.index
y_train_impute = train_impute['country_destination'].values
X_train_impute = train_impute.drop(labels='country_destination',axis=1).values

X_test_impute = X_test_impute.values

param_map = {'num_class': 12,
             'max_depth': 2,
             'eta': .1,  # We are actively tuning learning rate, or step size of the gradient
             'silent': 1,
             'objective': 'multi:softprob',
             'booster': 'gbtree',
             'gamma': 1,
             'min_child_weight': 1,
             'subsample': .5
             }

num_round = 2
dtrain = xgb.DMatrix(X_train_impute, label=y_train_impute)
dtest = xgb.DMatrix(X_test_impute)
mod = xgb.train(param_map, dtrain, num_round)

probs = mod.predict(dtest)
y_hat_xgboost = classify(probs)

submission = getSubmissionFile(user_idx=test_idx,predictions=y_hat_xgboost,k=5,country_map=country_label_df)
submission.to_csv("output/predictions/xgboost_raw_data_drop_na.csv")

