import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



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
        if isinstance(X,DataFrame):
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



    def fit_transform(self,X):
        """
        Impute all missing value of input data. Use classification or regression (depending on feature type)
        :param X: Input matrix (dataframe only)
        :return: Transformed version of X. All null values filled
        """

        # Check format of input data
        self.isDataframe(X)
        X_out = X.copy()

        # Print Null stats to Screen:
        self.printNullStatistics(X)

        # Get all rows with where any column (any j) is null
        rows_with_null = self.getNullSamples(X_out)
        null_indices = list(rows_with_null.index)
        for null_row_idx_i in null_indices:
            # Iterate over all samples with null values
            # We will fit a new model for each row (expensive!)
            tmp_row = rows_with_null.ix[null_row_idx_i]
            null_cols = list(tmp_row[tmp_row.isnull()].index)


            for i in range(len(null_cols)):
                # Iterate over null features
                # At each iteration, set target feature to impute with by model
                null_cols_to_remove = null_cols[:]
                null_cols_to_remove.pop(i)
                # Save column name
                y_name = null_cols[i]
                # Row to impute --> Test data
                test_row = tmp_row.drop(labels=null_cols,axis=0)
                # remove row to impute from training data
                full_col_train = X.drop(labels=null_row_idx_i,axis=0)
                # Collect training data. All other rows with complete data for feature to impute
                full_col_train = full_col_train[full_col_train[y_name].isnull() == False]
                # Drop any other features where data in row to impute is missing
                # This allows us to learn a model
                full_col_train.drop(labels = null_cols_to_remove, axis = 1,inplace=True)
                # Remove any additional missing data
                full_col_train.dropna(inplace=True)

                # Create training X and y to impute missing data
                impute_train_y = full_col_train[y_name]
                impute_train_X = full_col_train.drop(labels = y_name,axis=1)

                # Sample minibatch to reduce computational cost
                if self.minibatch:
                    impute_train_X, impute_train_y = self.getRandomBatch(X=impute_train_X,y=impute_train_y)

                # Model choice conditional on type of feature:
                # e.g., when the target is categorical, learn classifier,
                # when target is real-valued, learn regressor
                # Boolean to test if cat. variables were passed to constructor
                has_categorical_vars = self.hasCategoricalVars()
                if has_categorical_vars and y_name in self.categorical_vars:
                    nn = KNeighborsClassifier()
                else:
                    nn = KNeighborsRegressor()
                # convert data to numpy array for prediction
                impute_train_X_arr = impute_train_X.values
                impute_train_y_arr = impute_train_y.values
                test_row_arr = np.transpose(test_row.values.reshape(-1,1))

                # Make prediction using test data (row to impute)
                nn.fit(X=impute_train_X_arr,y=impute_train_y_arr)
                row_y_hat = nn.predict(test_row_arr)[0]
                # fill  row i, feature j with predicted value
                tmp_row[y_name] = row_y_hat
            # replace original row, i, with new row contaning imputed values
            X_out.ix[null_row_idx_i] = tmp_row

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
        rows_with_null = self.getNullSamples(X)
        null_rows_idx = list(rows_with_null.index)

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
        # Reset index of sampled data. easier to work with in the imputation routine
        X_sample.reset_index(drop=True,inplace=True)

        if self.impute_missing_vals:
            print("-----Beginning Imputation Algorithm: %s-----" % self.imputation_method)
            # Impute misisng values using method from Imputer class
            X_out = self.fit_transform(X=X_sample)
        else:
            X_out = X_sample.copy()

        return X_out





