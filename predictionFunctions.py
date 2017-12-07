from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import sys
from datetime import datetime
import xgboost as xgb


def getAccuracy(y_hat,y_true):
    y_correct = np.where(y_hat == y_true,1,0)
    accuracy_pct = np.mean(y_correct)
    return accuracy_pct

def classify(probs,k=5):
    probs_sort = np.argsort(probs,axis=1)
    probs_k = probs_sort[:,-k:]
    y_hat = np.flip(probs_k,axis=1)
    return y_hat

def classifyTwoStage(probs_stage_one, probs_stage_two,k=5):
    all_probs = np.concatenate((probs_stage_two,probs_stage_one),axis=1)
    probs_sort = np.argsort(all_probs,axis=1)
    probs_k = probs_sort[:,-k:]
    y_hat = np.flip(probs_k,axis=1)
    return y_hat

def getSubmissionFile(user_idx,predictions,k,country_map):
    user_id_list = []
    for user in user_idx:
        user_id_list += [user]*k

    df = DataFrame(data=predictions.flatten(), index=user_id_list, columns=['country'])
    df.index.rename('id', inplace=True)
    df = pd.merge(df, country_map, how='left', left_on='country', right_index=True)
    df.drop('country_x', axis=1, inplace=True)
    df.columns = ['country']
    return df


def ndcg(y_hat,y_true):
    indices = np.array(range(1,len(y_hat) + 1))
    relevance = np.where(y_true == y_hat,1,0)

    dcg_k = (np.exp2(relevance) - 1) / np.log2(indices + 1)
    dcg = np.sum(dcg_k)

    return dcg

"""Metrics to compute the model performance."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer


def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    ground_truth = [1, 0, 2]
    predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


def learnStackingLayer(X_train_meta,
                       y_train,
                       X_test_meta,
                       n_folds):
    """
    - Train meta-classier (stacked layer)
    - Predictions from models are used as features
    - Use 5-fold CV to learn best penalty term
    - Use L2 penalty. Don't want to induce sparsity, just regularization"""

    model_param_space = range(1,12)
    meta_penalty_eval = np.zeros((n_folds, len(model_param_space)))

    print "CV to select best lambda for stacked layer..."
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=0703)
    fold_cnt = 0
    total_cnt = 0
    for train_index, test_index in k_fold.split(X_train_meta):

        X_train_meta_cv = X_train_meta[train_index]
        y_train_meta_cv = y_train[train_index]

        X_test_meta_cv = X_train_meta[test_index]
        y_test_meta_cv = y_train[test_index]

        lambda_cnt = 0
        for lam in model_param_space:
            start_model = datetime.now()

            #model_meta_cv = LogisticRegression(penalty='l2',
            #                                   C=lam,
            #                                   fit_intercept=True,
            #                                   multi_class='multinomial',
            #                                   solver='sag'
            #                                   )
            #model_meta_cv.fit(X=X_train_meta_cv, y=y_train_meta_cv)
            #meta_probs_cv = model_meta_cv.predict_proba(X=X_test_meta_cv)

            #beta = model_meta_cv.coef_
            #np.savetxt(fname='output/meta-logit-beta.txt',X=beta)

            xgb_param_map = {'num_class': 12,
                         'max_depth': 4,
                         'eta': .1,
                         'silent': 1,
                         'objective': 'multi:softprob',
                         'booster': 'gbtree',
                         'gamma': 2.0,
                         'min_child_weight': lam,
                         'subsample': .5
                         }
            num_round = 4

            dtrain = xgb.DMatrix(X_train_meta_cv, label=y_train_meta_cv)
            dtest = xgb.DMatrix(X_test_meta_cv)

            mod_meta_cv = xgb.train(xgb_param_map, dtrain, num_round)
            meta_probs_cv = mod_meta_cv.predict(dtest)

            model_time = datetime.now() - start_model

            eval_ndcg = ndcg_score(ground_truth=y_test_meta_cv, predictions=meta_probs_cv)
            meta_penalty_eval[fold_cnt, lambda_cnt] = eval_ndcg

            lambda_cnt += 1
            total_cnt += 1

            # Print progress to screen
            progress = np.round(total_cnt / float(n_folds * len(model_param_space)),4)
            print "progress: " + str(progress) + " time: " + str(model_time)

        fold_cnt += 1

    mean_eval_ndcg_meta = np.mean(meta_penalty_eval, axis=1)
    best_meta_lambda_idx = np.argmax(mean_eval_ndcg_meta)
    best_meta_lambda = model_param_space[best_meta_lambda_idx]

    print "Best lambda: " + str(best_meta_lambda)

    #model_meta = LogisticRegression(penalty='l2',
    #                                C=best_meta_lambda,
    #                                fit_intercept=True,
    #                                multi_class='multinomial',
    #                                solver='sag'
    #                                )
    #model_meta.fit(X=X_train_meta, y=y_train)
    #meta_probs = model_meta.predict_proba(X=X_test_meta)

    xgb_param_map = {'num_class': 12,
                     'max_depth': 4,
                     'eta': .1,
                     'silent': 1,
                     'objective': 'multi:softprob',
                     'booster': 'gbtree',
                     'gamma': 2.0,
                     'min_child_weight': best_meta_lambda,
                     'subsample': .5
                     }
    num_round = 4

    dtrain = xgb.DMatrix(X_train_meta, label=y_train)
    dtest = xgb.DMatrix(X_test_meta)

    model_meta = xgb.train(xgb_param_map, dtrain, num_round)
    meta_probs = model_meta.predict(dtest)

    y_hat_meta = classify(probs=meta_probs)

    return y_hat_meta


