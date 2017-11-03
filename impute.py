import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy.stats import mode



class Imputer():

    def __init__(self,n_neighbors=5,
                 imputation_method='k-nn',
                 print = True,
                 categorical_vars = None,
                 minibatch=False,
                 minibatch_size = .1):

        self.n_neighbors = n_neighbors
        self.imputation_method = imputation_method
        self.print = print
        self.categorical_vars = categorical_vars
        self.minibatch = minibatch
        self.minibatch_size = minibatch_size


    def isDataframe(self,X):
        """Make sure data passed to methods is a pandas DataFrame"""
        if isinstance(X,DataFrame) | isinstance(X,np.ndarray) :
            pass
        else:
            raise Exception("Imputer is currently only able to process pandas DataFrames. Please reformat.")

    def printNullStatistics(self,X):
        if self.print:
            print("--Features w/ Null Values (%)--")
            pct_null = X.isnull().sum() / float(len(X))
            print(pct_null[pct_null > 0])
            print("--------------------------------")

    def getNullSamples(self,X):
        """ Find row in matrix that contain ANY column with a null value"""
        rows_with_null = X[X.isnull().any(axis=1)]
        return rows_with_null

    def getRandomBatch(self,X,y):
        """Randomly sample n rows from input X (matrix) and y (vector).
        Return: Reduced matrix and vector
        """

        idx = X.index
        # Sample by taking permutation of index
        permute_idx = np.random.permutation(idx)

        # Check type of mini-batch size. If int, sample first n,
        if isinstance(self.minibatch_size, int):
            take_ran_idx = permute_idx[:self.minibatch_size]
        elif isinstance(self.minibatch_size, float):
            #  if float sample p percent of data
            take_n = self.minibatch_size * len(X)
            take_n_int = int(round(take_n))
            take_ran_idx = permute_idx[:take_n_int]
        else:
            raise Exception("Argument 'minibatch_size' must be float or int.")

        X_sample = X.ix[take_ran_idx]
        y_sample = y.ix[take_ran_idx]

        return X_sample, y_sample

    def hasCategoricalVars(self):
        """
        Test if categorical variables are passed to constructor. If valid, non-empty list is passed, return True,
        o/w return false
        :return: Boolean
        """

        if isinstance(self.categorical_vars,list):
            return True
        else:
            return False

    def getFreqValues(self,X,cat_var_idx):
        col_idx = range(X.shape[1])
        #num_var_idx = list(set(col_idx) - set(cat_var_idx))
        freq_vals = {}

        for i in col_idx:
            if i in cat_var_idx:
                freq_vals[i] = mode((X[:, i]), nan_policy='omit')[0][0]
            else:
                freq_vals[i] = np.nanmean(X[:,i])


        return freq_vals

    def getCategoricalIdx(self,X):
        if isinstance(X,DataFrame):
            columns = X.columns
            column_idx = range(X.shape[1])
            column_dict = dict(zip(columns,column_idx))

            cat_vars = []
            for i in self.categorical_vars:
                cat_vars.append(column_dict[i])
        else:
            cat_vars = self.categorical_vars

        return cat_vars



    def impute(self,X):

        self.isDataframe(X)
        has_categorical_vars = self.hasCategoricalVars()

        if has_categorical_vars:
            cat_vars = self.getCategoricalIdx(X)
        else:
            cat_vars = []

        X_out = X.values

        n_cols = float(X_out.shape[1])

        # Find columns with null values
        X_is_null = np.isnan(X_out)
        X_not_null = ~X_is_null
        rows_complete_idx = np.where((np.sum(X_not_null,axis=1) / n_cols) == 1.0 )[0]
        X_complete = X_out[rows_complete_idx]

        fill_test_na = self.getFreqValues(X_out,cat_var_idx=cat_vars)

        cols_null_cnt = np.sum(X_is_null,axis=0)
        null_col_idx = np.where(cols_null_cnt > 0)[0]
        for col in null_col_idx:
            impute_train_y = X_complete[:,col]
            impute_train_x = np.delete(X_complete,obj=col,axis=1)

            test_x_idx = np.where(X_is_null[:,col])[0]
            impute_test_x = X_out[test_x_idx,:]


            null_cols_to_fill = list(null_col_idx[:])
            col_to_del = np.where(null_cols_to_fill == col)[0][0]
            null_cols_to_fill.pop(col_to_del)

            for j in null_cols_to_fill:

                impute_test_x[:,j] = np.where(np.isnan(impute_test_x[:,j]),
                                              fill_test_na[j],
                                              impute_test_x[:, j])

            # create k-nn object
            if (has_categorical_vars) & (col in cat_vars):
                nn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            else:
                nn = KNeighborsRegressor(n_neighbors=self.n_neighbors)

            # Make prediction using test data (row to impute)
            nn.fit(X=impute_train_x, y=impute_train_y)
            impute_test_x = np.delete(impute_test_x, obj=col, axis=1)
            row_y_hat = nn.predict(impute_test_x)
            # fill  row i, feature j with predicted value
            X_out[test_x_idx,col] = row_y_hat

        return X_out

class SampleBagger(Imputer):

    def __init__(self,impute_missing_vals=True,
                 ratio_dense_sparce=(2,1),
                 n_neighbors=5,
                 imputation_method='k-nn',
                 print=True,
                 categorical_vars=None,
                 minibatch=False,
                 minibatch_size=.1
                 ):
        super().__init__(n_neighbors=n_neighbors,
                 imputation_method=imputation_method,
                 print = print,
                 categorical_vars = categorical_vars,
                 minibatch=minibatch,
                 minibatch_size = minibatch_size)
        self.ratio_dense_sparce = ratio_dense_sparce
        self.impute_missing_vals = impute_missing_vals

    def getSampleWeights(self):
        return self.ratio_dense_sparce[0], self.ratio_dense_sparce[1]

    def getNullRowInt(self,X):
        null_rows_all = np.where(X.isnull())
        null_rows_int_idx = null_rows_all[0]
        return null_rows_int_idx


    def printNullStatistics(self,X):
        if self.print:
            print("--Bootstrapped Data: Features w/ Null Values (%)--")
            pct_null = X.isnull().sum() / float(len(X))
            print(pct_null[pct_null > 0])
            print("--------------------------------")

    def genSample(self,X):
        """
        Generate boostrapped sampled of input data (X: DataFrame)
        Sampling scheme uses weight ratio passed to constructor (ratio_dense_sparce) to sample data
        After sample is complete, if impute_missing_vals = True, then impute missing values on resulting boostrapped
        data
        :param X: Input data matrix (DataFrame)
        :return: Boostrapped DataFrame with missing values imputed (if impute_missing_vals = True)
        """
        # Test type of X. Return error if not a DataFrame
        self.isDataframe(X)
        n_rows = X.shape[0]
        # Retrieve sample weights for dense and sparse samples
        dense_sample_weight, sparce_sample_weight = self.getSampleWeights()

        all_rows_index = X.index
        # get indices of rows with at least one null value
        null_rows_idx = self.getNullRowInt(X)

        # Initialize sample probability vector
        sample_probs = np.ones(n_rows)
        # Create mask for dense and sparse samples
        sparce_sample_mask = np.zeros(n_rows,dtype=bool)
        sparce_sample_mask[null_rows_idx] = True
        # multiply dense and sparse samples by respective weights
        sample_probs[sparce_sample_mask] *= sparce_sample_weight
        sample_probs[~sparce_sample_mask] *= dense_sample_weight
        # Normalize weights to sum to one
        sample_probs = sample_probs / np.sum(sample_probs)
        # Sample row indices with replacement using sample probability weights from above
        sample_rows = np.random.choice(all_rows_index,
                                       size=n_rows,
                                       replace=True,
                                       p=sample_probs)
        # Create new DataFrame using sampled row indices
        X_sample = X.ix[sample_rows]



        if self.impute_missing_vals:
            X_sample_index = np.array(X_sample.index).reshape(-1,1)

            X_sample.reset_index(drop=True, inplace=True)

            print("-----Beginning Imputation Algorithm: %s-----" % self.imputation_method)
            # Impute misisng values using method from Imputer class
            X_impute = self.impute(X=X_sample)
            # Reset index of sampled data. easier to work with in the imputation routine
            #X_out.index = X_sample_index
            X_out = np.concatenate((X_sample_index,X_impute),axis=1)
        else:
            X_out = X_sample.copy()


        return X_out





