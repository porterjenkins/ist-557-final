import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



class Imputer():

    def __init__(self,n_neighbors=5,imputation_method='k-nn'):
        self.n_neighbors = n_neighbors
        self.imputation_method = imputation_method


    def getNullSamples(self,X):
        print("--Features w/ Null Values (%)--")
        pct_null = X.isnull().sum() / float(len(X))
        print(pct_null[pct_null > 0])
        print("--------------------------------")
        rows_with_null = X[X.isnull().any(axis=1)]

        return rows_with_null

    def encodeCategoricalVars(self,X):
        X_clean = X.copy()
        if isinstance(X,Series):
            X_clean = X_clean.to_frame()
        encode_string = LabelEncoder()
        non_numeric_cols = list(X_clean.select_dtypes(include=[object]).columns)


        for col in non_numeric_cols:
            col_encoded = encode_string.fit_transform(X_clean[col])
            X_clean[col] = col_encoded

        return X_clean

    def fit_transform(self,X):
        rows_with_null = self.getNullSamples(X)
        null_indices = list(rows_with_null.index)
        for null_row_idx_i in null_indices:

            tmp_row = rows_with_null.ix[null_row_idx_i]
            null_cols = list(tmp_row[tmp_row.isnull()].index)

            print(tmp_row)

            for i in range(len(null_cols)):
                null_cols_to_remove = null_cols[:]
                null_cols_to_remove.pop(i)
                y_name = null_cols[i]
                y_dtype = X.dtypes[y_name]

                print("y: ", y_name)
                print('drop: ', null_cols_to_remove)

                test_row = tmp_row.drop(labels=null_cols,axis=0)

                full_col_train = X.drop(labels=0,axis=0)
                full_col_train = full_col_train[full_col_train[y_name].isnull() == False]
                full_col_train.drop(labels = null_cols_to_remove, axis = 1,inplace=True)
                # Remove any additional missing data
                full_col_train.dropna(inplace=True)

                impute_train_y = full_col_train[y_name]
                impute_train_X = full_col_train.drop(labels = y_name,axis=1)



                if y_dtype == np.float:
                    nn = KNeighborsRegressor()
                else:
                    string_to_int = LabelEncoder()
                    impute_train_y = string_to_int.fit_transform(impute_train_y)
                    nn = KNeighborsClassifier()


                # convert data to numpy array for prediction

                impute_train_X_arr = impute_train_X.values
                impute_train_y_arr = impute_train_y.values
                test_row_arr = np.transpose(test_row.values.reshape(-1,1))

                nn.fit(X=impute_train_X_arr,y=impute_train_y_arr)
                row_y_hat = nn.predict(test_row_arr)[0]

                tmp_row[y_name] = row_y_hat

            X.ix[null_row_idx_i] = tmp_row

        return X




