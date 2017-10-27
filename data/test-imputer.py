import numpy as np
from sklearn import datasets
from pandas import DataFrame
from impute import Imputer
from sklearn.metrics import mean_absolute_error

train = DataFrame(datasets.load_iris()['data'])
train.columns = ['x1','x2','x3','x4']

# test categorical variable
train['x5'] = np.random.randint(low=0,high=4,size = len(train))

null_row_idx = list(np.random.randint(low=0,high=train.shape[0], size=10))
#null_cols_idx = list(np.random.randint(low=0,high=2, size=10))
null_cols_idx = [0,1,4]


null_data_true = train.iloc[null_row_idx,null_cols_idx].values
print(null_data_true)

train.iloc[null_row_idx,null_cols_idx] = np.nan

null_vals_pct = train.isnull().sum() / float(len(train))


impute = Imputer(categorical_vars=['x5'])
iris_clean = impute.fit_transform(train)
print(iris_clean.isnull().sum()/len(iris_clean))

null_data_predicted = iris_clean.iloc[null_row_idx,null_cols_idx].values

mae = mean_absolute_error(y_true=null_data_true,y_pred=null_data_predicted)
print("Prediction error of imputation: ", mae)