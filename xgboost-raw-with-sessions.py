import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier

## Inverse Transform Label Encoder

countries = ['AU','CA','DE','ES','FR','GB','IT','NDF','NL','PT','US','other']
country_labels = dict(zip(range(len(countries)),countries))

country_label_df = DataFrame.from_dict(country_labels,orient='index')
country_label_df.columns = ['country']


def getAccuracy(y_hat,y_true):
    y_correct = np.where(y_hat == y_true,1,0)
    accuracy_pct = np.mean(y_correct)
    return accuracy_pct

def classify(probs):
    y_hat = np.argmax(probs,axis=1)
    return y_hat



### Imputed Data

train_impute = pd.read_csv("data/train_with_session.csv",index_col=0)
print train_impute.shape
X_test_impute = pd.read_csv("data/test_with_session.csv",index_col=0)

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


impute_data_out_df = DataFrame(data=y_hat_xgboost,index=test_idx,columns=['country'])
impute_data_out_df.index.rename('id',inplace=True)
impute_data_out_df = pd.merge(impute_data_out_df,country_label_df,how='left',left_on='country',right_index=True)
impute_data_out_df.drop('country_x',axis=1,inplace=True)
impute_data_out_df.columns = ['country']


impute_data_out_df.to_csv("output/predictions/xgboost_raw_session.csv")
