import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



class Imputer():

    def __init__(self,n_neighbors=5,imputation_method='k-nn',print = True,categorical_vars = None):
        self.n_neighbors = n_neighbors
        self.imputation_method = imputation_method
        self.print = print
        self.categorical_vars = categorical_vars


    def isDataframe(self,X):
        if isinstance(X,DataFrame):
            pass
        else:
            raise Exception("Imputer is currently only able to process pandas DataFrame. Please reformat.")


    def getNullSamples(self,X):
        if self.print:
            print("--Features w/ Null Values (%)--")
            pct_null = X.isnull().sum() / float(len(X))
            print(pct_null[pct_null > 0])
            print("--------------------------------")
        rows_with_null = X[X.isnull().any(axis=1)]

        return rows_with_null


    def fit_transform(self,X):

        # Check format of input data
        self.isDataframe(X)

        # Check for categorical variables

        rows_with_null = self.getNullSamples(X)
        null_indices = list(rows_with_null.index)
        for null_row_idx_i in null_indices:

            tmp_row = rows_with_null.ix[null_row_idx_i]
            null_cols = list(tmp_row[tmp_row.isnull()].index)


            for i in range(len(null_cols)):
                null_cols_to_remove = null_cols[:]
                null_cols_to_remove.pop(i)
                y_name = null_cols[i]
                y_dtype = X.dtypes[y_name]


                test_row = tmp_row.drop(labels=null_cols,axis=0)

                full_col_train = X.drop(labels=0,axis=0)
                full_col_train = full_col_train[full_col_train[y_name].isnull() == False]
                full_col_train.drop(labels = null_cols_to_remove, axis = 1,inplace=True)
                # Remove any additional missing data
                full_col_train.dropna(inplace=True)

                impute_train_y = full_col_train[y_name]
                impute_train_X = full_col_train.drop(labels = y_name,axis=1)



                if y_name in self.categorical_vars:
                    nn = KNeighborsClassifier()
                    print("%s - classifier..." % y_name)
                else:
                    nn = KNeighborsRegressor()
                    print("%s - regressor..." % y_name)


                # convert data to numpy array for prediction

                impute_train_X_arr = impute_train_X.values
                impute_train_y_arr = impute_train_y.values
                test_row_arr = np.transpose(test_row.values.reshape(-1,1))

                nn.fit(X=impute_train_X_arr,y=impute_train_y_arr)
                row_y_hat = nn.predict(test_row_arr)[0]

                tmp_row[y_name] = row_y_hat

            X.ix[null_row_idx_i] = tmp_row

        return X




