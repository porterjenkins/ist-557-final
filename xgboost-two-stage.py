import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from predictionFunctions import *
import matplotlib.pyplot as plt
from scipy.stats import itemfreq

## Inverse Transform Label Encoder

countries_two_stage = ['AU','CA','DE','ES','FR','GB','IT','NL','PT','US','other','NDF']
country_labels = dict(zip(range(len(countries_two_stage)),countries_two_stage))

country_label_df = DataFrame.from_dict(country_labels,orient='index')
country_label_df.columns = ['country']



train = pd.read_csv("data/train_with_session.csv",index_col=0)
X_test = pd.read_csv("data/test_with_session.csv",index_col=0)

test_idx = X_test.index
y_train = train['country_destination'].values
X_train = train.drop(labels='country_destination',axis=1).values
X_test = X_test.values
dtest = xgb.DMatrix(X_test)

# Building two-stage classifier

# First stage: Classify to 'NDF' or other
# Use all data available
y_train_stage_one = np.where(y_train == 7, 1, 0)
param_map_stage_one = {'num_class': 2,
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
dtrain_stage_one = xgb.DMatrix(X_train, label=y_train_stage_one)
mod_stage_one = xgb.train(param_map_stage_one, dtrain_stage_one, num_round)
probs_stage_one = mod_stage_one.predict(dtest)


# Second stage: Ignore "NDF" values, classify to valid country
# Use only samples whose 'country_destination' is a valid country, i.e., not equal to 'NDF'
train_idx = np.where(y_train !=7)
X_train_stage_two = X_train[train_idx]
y_train_stage_two = y_train[train_idx]

y_train_stage_two = np.where(y_train_stage_two > 7, y_train_stage_two - 1, y_train_stage_two)

param_map_stage_two = {'num_class': 11,
             'max_depth': 2,
             'eta': .1,
             'silent': 1,
             'objective': 'multi:softprob',
             'booster': 'gbtree',
             'gamma': 1,
             'min_child_weight': 1,
             'subsample': .5
             }

num_round = 2
dtrain_stage_two = xgb.DMatrix(X_train_stage_two, label=y_train_stage_two)
dtest_stage_two = xgb.DMatrix(X_train_stage_two)
mod_stage_two = xgb.train(param_map_stage_two, dtrain_stage_two, num_round)
probs_stage_two = mod_stage_two.predict(dtest)


# Make top five predictions from two-stage classifier
probs_stage_one_vector = probs_stage_one[:,1].reshape(-1,1)
y_hat_xgboost = classifyTwoStage(probs_stage_one=probs_stage_one_vector,probs_stage_two=probs_stage_two,k=5)

# Write predictions to file
submission = getSubmissionFile(user_idx=test_idx,predictions=y_hat_xgboost,k=5,country_map=country_label_df)
submission.to_csv("output/predictions/xgboost_two_stage.csv")

















