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

### Imputed Data

train_impute = pd.read_csv("data/rawuser-train.csv",index_col=0)
X_test_impute = pd.read_csv("data/impute-user-test.csv",index_col=0)


test_idx = X_test_impute.index
y_train_impute = train_impute['country_destination'].values
X_train_impute = train_impute.drop(labels='country_destination',axis=1).values


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X=X_train_impute,y=y_train_impute)

rf_y_hat = rf.predict(X=X_test_impute)


impute_data_out_df = DataFrame(data=rf_y_hat,index=test_idx,columns=['country'])
impute_data_out_df.index.rename('id',inplace=True)
impute_data_out_df = pd.merge(impute_data_out_df,country_label_df,how='left',left_on='country',right_index=True)
impute_data_out_df.drop('country_x',axis=1,inplace=True)
impute_data_out_df.columns = ['country']


impute_data_out_df.to_csv("output/predictions/rf_impute_data.csv")








