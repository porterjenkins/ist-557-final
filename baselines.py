import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def getAccuracy(y_hat,y_true):
    y_correct = np.where(y_hat == y_true,1,0)
    accuracy_pct = np.mean(y_correct)
    return accuracy_pct

def classify(probs):
    y_hat = np.argmax(probs,axis=1)
    return y_hat




user_raw = pd.read_csv("data/raw-user-train.csv",index_col=0)
user_raw_no_na = user_raw.dropna()
#print "Naive Baseline: ", user_raw['country_destination'].value_counts() / float(len(user_raw))

y = user_raw['country_destination'].values
X = user_raw.drop(labels='country_destination',axis=1).values

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X,y,test_size=.2,random_state=0703)


y_no_na = user_raw_no_na['country_destination'].values
X_no_na = user_raw_no_na.drop(labels='country_destination',axis=1).values

X_train_raw_no_na, X_test_raw_no_na, y_train_raw_no_na, y_test_raw_no_na = train_test_split(X_no_na,y_no_na,test_size=.2,random_state=0703)


# XBGOOST Drop NA

dtrain_no_na = xgb.DMatrix(X_train_raw_no_na, label=y_train_raw_no_na)
dtest_no_na = xgb.DMatrix(X_test_raw_no_na, label=y_test_raw_no_na)


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
mod = xgb.train(param_map, dtrain_no_na, num_round)


# make prediction
probs = mod.predict(dtest_no_na)

y_hat_xgboost_no_na = classify(probs)
xgboost_accuracy = getAccuracy(y_hat=y_hat_xgboost_no_na,y_true=y_test_raw_no_na)
print 'Raw Data Accuracy, with No NA (XGBOOST)', xgboost_accuracy

# XBOOST with NA
dtrain = xgb.DMatrix(X_train_raw, label=y_train_raw)
dtest = xgb.DMatrix(X_test_raw, label=y_test_raw)

mod = xgb.train(param_map, dtrain, num_round)


# make prediction
probs = mod.predict(dtest)

y_hat_xgboost = classify(probs)
xgboost_accuracy = getAccuracy(y_hat=y_hat_xgboost,y_true=y_test_raw)
print 'Raw Data Accuracy, with with NA (XGBOOST)', xgboost_accuracy



# XGBOOST Impute

user_impute = pd.read_csv("data/impute-user-train.csv",index_col=0)


y = user_impute['country_destination'].values
X = user_impute.drop(labels='country_destination',axis=1).values

X_train_impute, X_test_impute, y_train_impute, y_test_impute = train_test_split(X,y,test_size=.2,random_state=0524)
dtrain = xgb.DMatrix(X_train_impute, label=y_train_impute)
dtest = xgb.DMatrix(X_test_impute, label=y_test_impute)
mod = xgb.train(param_map, dtrain, num_round)


probs = mod.predict(dtest)

y_hat_xgboost = classify(probs)
xgboost_accuracy = getAccuracy(y_hat=y_hat_xgboost,y_true=y_test_impute)
print 'Impute Data Accuracy (XGBOOST)', xgboost_accuracy


## Random Forest drop na

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X=X_train_raw_no_na,y=y_train_raw_no_na)

rf_y_hat = rf.predict(X=X_test_raw_no_na)

rf_accuracy = getAccuracy(y_hat=rf_y_hat,y_true=y_test_raw_no_na)
print 'Raw Data Accuracy, no NA (Random Forest)', rf_accuracy


# Random Forest Impute

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X=X_train_impute,y=y_train_impute)

rf_y_hat = rf.predict(X=X_test_impute)
rf_accuracy = getAccuracy(y_hat=rf_y_hat,y_true=y_test_impute)
print 'Raw Data Accuracy, Impute (Random Forest)', rf_accuracy