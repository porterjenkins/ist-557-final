import numpy as np
from impute import Imputer, SampleBagger
from pandas import DataFrame
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocesser:

    def __init__(self,bagging):
        self.bagging = bagging

    def encodeCols(self,X, columns_to_encode):
        """
        Iterates over given columns of DataFrame and converts each to ints. We use a level-encoder, not one hot encoding.
        For example:
            If, x_i = ['male,'male','female'] becomes x_i_clean = [0,0,1]
        :param X: DataFrame
        :param columns_to_encode: list of columns to encode (list)
        :return:
        """
        X_out = X.copy()

        label_encoder = LabelEncoder()

        for col in columns_to_encode:
            X_out[col] = X_out[col].astype(str)
            X_out[col] = label_encoder.fit_transform(X_out[col])
            # Should we save mapping dicts to file here?

        return X_out

    def transform(self,train):

        if isinstance(train,DataFrame):
            pass
        else:
            raise Exception("X must be a pandas DataFrame")

        # the age variable looks a bit off (i.e., some people entered the year they were born instead of their age),
        # so I'm going to follow some pre-processing steps from
        # https://github.com/davidgasquez/kaggle-airbnb/blob/master/notebooks/Preprocessing.ipynb

        user_with_year_age_mask = train['age'] > 1000
        train.loc[user_with_year_age_mask, 'age'] = 2015 - train.loc[user_with_year_age_mask, 'age']

        train.loc[(train['age'] > 100) | (train['age'] < 18), 'age'] = -1

        # modifying date information
        # again, using code from https://github.com/davidgasquez/kaggle-airbnb/blob/master/notebooks/Preprocessing.ipynb
        train['date_account_created'] = pd.to_datetime(train['date_account_created'], errors='ignore')
        train['date_first_active'] = pd.to_datetime(train['timestamp_first_active'], format='%Y%m%d%H%M%S')

        # Convert to DatetimeIndex:
        date_account_created = pd.DatetimeIndex(train['date_account_created'])
        date_first_active = pd.DatetimeIndex(train['date_first_active'])

        # In[12]:

        # split dates into day, week, month, year:
        train['day_account_created'] = date_account_created.day
        train['weekday_account_created'] = date_account_created.weekday
        train['week_account_created'] = date_account_created.week
        train['month_account_created'] = date_account_created.month
        train['year_account_created'] = date_account_created.year
        train['day_first_active'] = date_first_active.day
        train['weekday_first_active'] = date_first_active.weekday
        train['week_first_active'] = date_first_active.week
        train['month_first_active'] = date_first_active.month
        train['year_first_active'] = date_first_active.year

        # In[13]:

        # Get the difference(time lag) between the date in which the account was created and when it was first active:
        train['time_lag'] = (date_account_created.values - date_first_active.values).astype(int)

        # In[14]:

        # drop duplicated columns
        drop_list = [
            'date_account_created',
            'date_first_active',
            'timestamp_first_active'
        ]

        train.drop(drop_list, axis=1, inplace=True)

        # making sure categorical variables are categorical
        train["gender"] = train["gender"].astype('category')
        train["signup_method"] = train["signup_method"].astype('category')
        train["language"] = train["language"].astype('category')
        train["affiliate_channel"] = train["affiliate_channel"].astype('category')
        train["affiliate_provider"] = train["affiliate_provider"].astype('category')
        train["first_affiliate_tracked"] = train["first_affiliate_tracked"].astype('category')
        train["signup_app"] = train["signup_app"].astype('category')
        train["first_device_type"] = train["first_device_type"].astype('category')
        train["first_browser"] = train["first_browser"].astype('category')

        if "country_destination" in train.columns:
            train["country_destination"] = train["country_destination"].astype('category')

        train_dtypes = train.dtypes
        numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numeric_mask = np.isin(train_dtypes, numeric_types)

        numeric_cols = list(train_dtypes[numeric_mask].index)
        nonnumeric_cols = list(train_dtypes[~numeric_mask].index)

        train_encoded = self.encodeCols(train, nonnumeric_cols)

        null_vals_pct = train_encoded.isnull().sum() / float(len(train_encoded))
        print("Null values before imputation")
        print(null_vals_pct[null_vals_pct > 0])


        if self.bagging:
            fill_na = SampleBagger(impute_missing_vals=True,
                                   ratio_dense_sparce=(5, 1),
                                   n_neighbors=5,
                                   print=True,
                                   categorical_vars=nonnumeric_cols,
                                   minibatch=True,
                                   minibatch_size=.1)
            train_out = DataFrame(fill_na.genSample(X=train_encoded))
        else:
            fill_na = Imputer(n_neighbors=5,
                                   print=True,
                                   categorical_vars=nonnumeric_cols,
                                   minibatch=True,
                                   minibatch_size=.1)
            train_out = DataFrame(fill_na.impute(train_encoded))


        null_vals_pct = train_out.isnull().sum() / float(len(train_out))
        print("Null values after imputation")
        print(null_vals_pct)


    def transform_gender(self,gender_data):
        """
        Preprocessing function for gender data
        :param gender_data: gender data (DataFrame)
        :return: transformed gender data (DataFrame)
        """

        pass

    def transform_log(self, log_data):
        """
        Preprocessing function for LOG data
        :param log_data: log data (DataFrame)
        :return: transformed l0g data (DataFrame)
        """
        if isinstance(train,DataFrame):
            pass
        else:
            raise Exception("X must be a pandas DataFrame")

        #descriptives for action, i.e., counting the total number of different actions.
        total_actions = sessions['action'].value_counts()

        #descriptives by user.
        row_count = sessions.groupby('user_id').count()

        #total seconds elapsed per user.
        total_sec = sessions.groupby('user_id')[['secs_elapsed']].sum()

        #creating data set of new features that I will continually add to.
        new_features = pandas.concat([row_count, total_sec], axis=1)

        #average seconds elapsed per user.
        avg_sec = sessions.groupby('user_id')[['secs_elapsed']].mean()

        #adding feature to new_features.
        new_features = pandas.concat([new_features, avg_sec], axis=1)

        #rename columns so there isn't any overlap.
        new_features.columns.values[5] = 'total_secs'
        new_features.columns.values[6] = 'avg_secs'

        #frequency of action by user.
        action_freq = pandas.DataFrame({'action_count' : sessions.groupby( ["user_id", "action"] ).size()}).reset_index()

        #change action_freq from long to wide.
        action_wide = action_freq.pivot(index = 'user_id', columns = 'action', values = 'action_count')

        #add action type info to new_features data set.
        new_features = pandas.concat([new_features, action_wide], axis=1)

        #frequency of action detail by user.
        action_detail_freq = pandas.DataFrame({'action_detail_count' : sessions.groupby( ["user_id", "action_detail"] ).size()}).reset_index()

        #change action_detail_freq from long to wide.
        action_detail_wide = action_detail_freq.pivot(index = 'user_id', columns = 'action_detail', values = 'action_detail_count')

        #add action type info to new_features data set
        new_features = pandas.concat([new_features, action_detail_wide], axis=1)

        #fill all NAs in new_features with zeros (since all features are frequencies)
        new_features1 = new_features.fillna(0)
        print(new_features1)



        




