import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from predictionFunctions import *
from sklearn.model_selection import KFold
import itertools

## Inverse Transform Label Encoder

countries = ['AU','CA','DE','ES','FR','GB','IT','NDF','NL','PT','US','other']
country_labels = dict(zip(range(len(countries)),countries))

country_label_df = DataFrame.from_dict(country_labels,orient='index')
country_label_df.columns = ['country']

### Import data

train = pd.read_csv("data/train_with_session-11-16.csv",index_col=0)
X_test = pd.read_csv("data/test_with_session-11-16.csv",index_col=0)

# Reduce size of training data:

train = train.sample(frac=.25)

test_idx = X_test.index

y_train = train['country_destination'].values
X_train = train.drop(labels='country_destination',axis=1).values

X_test = X_test.values



# Create KFold object
n_folds = 3
k_fold = KFold(n_splits=n_folds,shuffle=True,random_state=0703)
# Create hypercube for grid search
param_list = ['eta','gamma','max_depth','min_child_weight','num_round']
param_hypercube = {
    'eta': np.array([.1]),
    'gamma':np.array([1.0,1.5,2.5,3.0,3.5]),
    'max_depth': np.array([4,5,6,7]),
    'min_child_weight': np.array([8,9,10,12,15,20]),
    'num_round': np.array([4,5,6,7])
}

hypercube_dim = {
    'eta': len(param_hypercube['eta']),
    'gamma': len(param_hypercube['gamma']),
    'max_depth': len(param_hypercube['max_depth']),
    'min_child_weight': len(param_hypercube['min_child_weight']),
    'num_round': len(param_hypercube['num_round'])


}


dim_cv_tensor = (n_folds,
                 hypercube_dim['eta'],
                 hypercube_dim['gamma'],
                 hypercube_dim['max_depth'],
                 hypercube_dim['min_child_weight'],
                 hypercube_dim['num_round'])


cv_tensor = np.zeros(shape=dim_cv_tensor)

#for comb in param_cartesian:
#    print comb



fold_cntr = 0
for train_index, test_index in k_fold.split(X_train):

    # Split folds: training set, and test set (fold)
    X_train_cv = X_train[train_index]
    y_train_cv = y_train[train_index]

    X_validation_cv = X_train[test_index]
    y_validation_cv = y_train[test_index]

    # Convert folds to libsvm format
    dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
    dtest = xgb.DMatrix(X_validation_cv)

    param_cartesian = itertools.product(param_hypercube['eta'],
                                        param_hypercube['gamma'],
                                        param_hypercube['max_depth'],
                                        param_hypercube['min_child_weight'],
                                        param_hypercube['num_round'])

    for combination in param_cartesian:
        print combination
        eta = combination[0]
        gamma = combination[1]
        max_depth = combination[2]
        min_child_weight = combination[3]
        num_round = combination[4]

        eta_idx = np.where(param_hypercube['eta'] == eta)[0][0]
        gamma_idx = np.where(param_hypercube['gamma'] == gamma)[0][0]
        max_depth_idx = np.where(param_hypercube['max_depth'] == max_depth)[0][0]
        min_child_weight_idx = np.where(param_hypercube['min_child_weight'] == min_child_weight)[0][0]
        num_round_idx = np.where(param_hypercube['num_round'] == num_round)[0][0]



        param_map = {'num_class': 12,
                     'max_depth': max_depth,
                     'eta': eta,
                     'silent': 1,
                     'objective': 'multi:softprob',
                     'booster': 'gbtree',
                     'gamma': gamma,
                     'min_child_weight': min_child_weight,
                     'subsample': .5
                     }


        mod = xgb.train(param_map, dtrain,num_round)
        probs = mod.predict(dtest)
        y_hat_xgboost = classify(probs)

        eval_ndcg = ndcg_score(ground_truth=y_validation_cv,predictions=probs)
        cv_tensor[fold_cntr,
                  eta_idx,
                  gamma_idx,
                  max_depth_idx,
                  min_child_weight_idx,
                  num_round_idx] = eval_ndcg

    fold_cntr += 1
    print "-------END FOLD---------"


k_fold_ndcg = np.mean(cv_tensor,axis=0)
max_ndcg = np.amax(k_fold_ndcg)
argmax_ndcg = np.where(k_fold_ndcg == max_ndcg)




best_params_idx = dict(zip(param_list,[x[0] for x in argmax_ndcg]))

f = open('output/xgboost-tune.txt','w')
f.write("Max NDCG value: %s \n" % max_ndcg)

for key in best_params_idx.keys():
    #print key, param_hypercube[key][best_params_idx[key]]
    f.write(key + ": " + str(param_hypercube[key][best_params_idx[key]]))
    f.write('\n')










