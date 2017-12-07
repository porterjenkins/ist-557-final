import numpy as np
from predictionFunctions import *


## Inverse Transform Label Encoder

countries = ['AU','CA','DE','ES','FR','GB','IT','NDF','NL','PT','US','other']
country_labels = dict(zip(range(len(countries)),countries))

country_label_df = DataFrame.from_dict(country_labels,orient='index')
country_label_df.columns = ['country']

# Import numpy arrays

X_train_meta = np.load(file='output/x-train-meta.npy')
X_test_meta = np.load(file='output/x-test-meta.npy')
y_train = np.load(file='output/y-train.npy')
test_idx = np.load(file='output/test-idx.npy')

# Concatenate full data set to predictions

train_full = pd.read_csv("data/train_with_session_language_fill_all_nan.csv",index_col=0)
X_test_full = pd.read_csv("data/test_with_session_language_fill_all_nan.csv",index_col=0).values
X_train_full = train_full.drop(labels='country_destination',axis=1).values

X_train_meta = np.concatenate((X_train_meta,X_train_full),axis=1)
X_test_meta = np.concatenate((X_test_meta,X_test_full),axis=1)


#y_hat_meta = learnStackingLayer(X_train_meta=X_train_meta,
#                                y_train=y_train,
#                                X_test_meta=X_test_meta,
#                                n_folds=5)
n_estimators = 1000
param_map = {'num_class': 12,
             'max_depth': 4,
             'eta': .1,
             'silent': 1,
             'objective': 'multi:softprob',
             'booster': 'gbtree',
             'gamma': 2.0,
             'min_child_weight': 5,
             'subsample': .5,
             'n_estimators': n_estimators
             }

num_round = 4
dtrain = xgb.DMatrix(X_train_meta, label=y_train)
dtest = xgb.DMatrix(X_test_meta)
mod = xgb.train(param_map, dtrain, num_round)

probs = mod.predict(dtest)
y_hat_meta = classify(probs)


submission = getSubmissionFile(user_idx=test_idx,predictions=y_hat_meta,k=5,country_map=country_label_df)
submission.to_csv("output/predictions/stacked-classifiers.csv")

print n_estimators