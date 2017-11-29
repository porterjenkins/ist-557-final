from pandas import DataFrame
import pandas as pd
import numpy as np


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




