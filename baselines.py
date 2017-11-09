import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
import numpy as np


def getAccuracy(y_hat,y_true):
    y_correct = np.where(y_hat == y_true,1,0)
    accuracy_pct = np.mean(y_correct)
    return accuracy_pct

def xgboostClassify(probs):
    y_hat = np.argmax(probs,axis=1)
    return y_hat




user_raw = pd.read_csv("data/raw-user-train.csv",index_col=0)

y = user_raw['country_destination'].values
X = user_raw.drop(labels='country_destination',axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0703)

n_class = np.max(y)



dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


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
mod = xgb.train(param_map, dtrain, num_round)


# make prediction
probs = mod.predict(dtest)

y_hat_xgboost = xgboostClassify(probs)
xgboost_accuracy = getAccuracy(y_hat=y_hat_xgboost,y_true=y_test)
print 'Raw Data Accuracy', xgboost_accuracy





user_impute = pd.read_csv("data/impute-user-train.csv",index_col=0)
#user_raw.dropna(inplace=True)

y = user_impute['country_destination'].values
X = user_impute.drop(labels='country_destination',axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0703)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
mod = xgb.train(param_map, dtrain, num_round)


probs = mod.predict(dtest)

y_hat_xgboost = xgboostClassify(probs)
xgboost_accuracy = getAccuracy(y_hat=y_hat_xgboost,y_true=y_test)
print 'Impute Data Accuracy', xgboost_accuracy
