import numpy as np
from sklearn import datasets
from pandas import DataFrame
from impute import Imputer

train = DataFrame(datasets.load_iris()['data'])
train.columns = ['x1','x2','x3','x4']

null_row = list(np.random.randint(low=0,high=train.shape[0], size=10))
null_cols = list(np.random.randint(low=0,high=2, size=10))

train.iloc[null_row,null_cols] = np.nan

null_vals_pct = train.isnull().sum() / float(len(train))



impute = Imputer()
iris_clean = impute.fit_transform(train)

print(iris_clean.isnull().sum()/len(iris_clean))
