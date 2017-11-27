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

