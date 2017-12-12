# Note: This is a Python 2.7 file
# This file learns meta-classifiers and constructs meta-data set filled with top 5 predicted destinations for each model

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import KFold
from predictionFunctions import *
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys
from datetime import datetime


# Xgboost parameter mapping

param_map = {'num_class': 12,
             'max_depth': 4,
             'eta': .1,
             'silent': 1,
             'objective': 'multi:softprob',
             'booster': 'gbtree',
             'gamma': 2.0,
             'min_child_weight': 10,
             'subsample': .5
             }
num_round = 4

### Read in data

train = pd.read_csv("data/train_with_session_language_fill_all_nan.csv",index_col=0)
X_test = pd.read_csv("data/test_with_session_language_fill_all_nan.csv",index_col=0)

test_idx = X_test.index
y_train = train['country_destination'].values
X_train = train.drop(labels='country_destination',axis=1)


X_train_dense = X_train.dropna(axis=1,how='any')
X_test_dense = X_test.dropna(axis=1,how='any')

X_test = X_test.values
X_train = X_train.values
X_train_dense = X_train_dense.values
X_test_dense = X_test_dense.values


# Create meta data sets
n_models = 3
n_predictions = 5
n_train = X_train.shape[0]
n_test = X_test.shape[0]

X_train_meta = np.zeros((n_train,n_models*n_predictions))
X_test_meta = np.zeros((n_test,n_models*n_predictions))


# Create training folds
n_folds = 5
k_fold = KFold(n_splits=n_folds,shuffle=True,random_state=0703)

fold_cnt = 0
for train_index, test_index in k_fold.split(X_train):
    sys.stdout.write("\r Test Fold: %s \n" % (fold_cnt))
    #sys.stdout.flush()

    X_train_cv = X_train[train_index]
    y_train_cv = y_train[train_index]

    X_train_cv_dense = X_train_dense[train_index]
    y_train_cv_dense = y_train[train_index]

    X_validation_cv = X_train[test_index]
    y_validation_cv = y_train[test_index]

    X_validation_cv_dense = X_train_dense[test_index]
    y_validation_cv_dense = y_train[test_index]


    dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
    dtest = xgb.DMatrix(X_validation_cv)

    ## Train xgboost

    sys.stdout.write("Training Model: XGBoost\n")
    #sys.stdout.flush()
    mod = xgb.train(param_map, dtrain, num_round)

    # Train random forest
    sys.stdout.write("Training Model: RandomForest\n")
    #sys.stdout.flush()
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X=X_train_cv_dense, y=y_train_cv_dense)

    # Train lasso
    sys.stdout.write("Training Model: Logit\n")
    sys.stdout.flush()
    logit = LogisticRegression(penalty='L1',
                               C=1,
                               fit_intercept=True,
                               solver='saga',
                               multi_class='multinomial',
                               max_iter=10
                               )
    logit.fit(X=X_train_cv_dense, y=y_train_cv_dense)

    # Make predictions
    probs_xgboost = mod.predict(dtest)
    y_hat_xgboost  = classify(probs_xgboost)

    probs_rf = rf.predict_proba(X_validation_cv_dense)
    y_hat_rf = classify(probs_rf)

    probs_logit = logit.predict_proba(X=X_validation_cv_dense)
    y_hat_logit = classify(probs_logit)

    X_train_meta[test_index,0:5] = y_hat_xgboost
    X_train_meta[test_index,5:10] = y_hat_rf
    X_train_meta[test_index,10:15] = y_hat_logit

    fold_cnt += 1

print "Cross-validation complete...."
# Fit all three models to full training data to populate test data for stacked layer

print "Training on full data...."
d_train_all_data= xgb.DMatrix(X_train, label=y_train)
d_test_all_data = xgb.DMatrix(X_test)

mod_all_data = xgb.train(param_map, d_train_all_data, num_round)

# Train random forest
rf_all_data = RandomForestClassifier(n_estimators=100)
rf_all_data.fit(X=X_train_dense, y=y_train)

# Train lasso
logit_all_data = LogisticRegression(penalty='L1',
                           C=1,
                           fit_intercept=True,
                           solver='saga',
                           multi_class='multinomial',
                           max_iter=10
                           )
logit_all_data.fit(X=X_train_dense, y=y_train)

# Make predictions
probs_xgboost_all_data = mod_all_data.predict(d_test_all_data)
y_hat_xgboost_all_data = classify(probs_xgboost_all_data)

probs_rf_all_data = rf_all_data.predict_proba(X_test_dense)
y_hat_rf_all_data = classify(probs_rf_all_data)

probs_logit_all_data = logit_all_data.predict_proba(X=X_test_dense)
y_hat_logit_all_data = classify(probs_logit_all_data)

X_test_meta[:, 0:5] = y_hat_xgboost_all_data
X_test_meta[:, 5:10] = y_hat_rf_all_data
X_test_meta[:, 10:15] = y_hat_logit_all_data

# Save meta data (training, test)
np.save(file='output/x-train-meta',arr=X_train_meta)
np.save(file='output/x-test-meta',arr=X_test_meta)
np.save(file='output/y-train',arr=y_train)
np.save(file='output/test-idx',arr=test_idx)

print "Complete!"
