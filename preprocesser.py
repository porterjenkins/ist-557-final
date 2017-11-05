import numpy as np
from impute import Imputer, SampleBagger
from pandas import DataFrame
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocesser:

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

    def transform_user(self,train,bagging=True):

        if isinstance(train,DataFrame):
            pass
        else:
            raise Exception("X must be a pandas DataFrame")

        # Drop field 'date_first_booking', since we are predicting the first booking. We will never have access to this
        # we observing new data

        train.drop('date_first_booking',axis=1,inplace=True)


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


        if bagging:
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

        return train_out


    def transform_gender(self,age_gender):
        """
        Preprocessing function for gender data
        :param gender_data: gender data (DataFrame)
        :return: transformed gender data (DataFrame)
        """

        # Create function to compute percentage in group by
        def norm_population(x):
            return x / np.sum(x)

        ## two group by objects. One for sum, the other for percentages

        # Group by country, age tier, gender and compute sum
        country_totals = age_gender.groupby(['country_destination', 'age_bucket', 'gender']).agg('sum')
        # Group by country, age tier, gender and compute percent
        country_pct = country_totals.groupby(level=0).apply(norm_population)
        country_pct.reset_index(inplace=True)

        # Split males and female for pivot operation
        # We will later flatten data into one dataframe for output

        male = country_pct[country_pct['gender'] == 'male']
        male.name = 'males'
        female = country_pct[country_pct['gender'] == 'male']
        female.name = 'females'

        # Put split df's by gender into list
        gender_df_list = [male, female]
        df_pivot_clean = []
        # iterate over gender df's
        for df in gender_df_list:
            # Pivot operation: Transform age_bucket rows into colums
            df_pivot = df.pivot(index='age_bucket', columns='country_destination', values='population_in_thousands')
            df_cols = df_pivot.columns
            # Rename columns to include gender prefix
            df_cols_gender = [df.name + "_" + x for x in df_cols]
            df_pivot.columns = df_cols_gender
            # Add dataframe to list
            df_pivot_clean.append(df_pivot)

        # combine data frames. Concetenate over rows (by columns)
        out = pd.concat(df_pivot_clean, axis=1)
        return out

    def transform_log(self, sessions):
        """
        Preprocessing function for LOG data
        :param log_data: log data (DataFrame)
        :return: transformed l0g data (DataFrame)
        """
        if isinstance(sessions,DataFrame):
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
        new_features = pd.concat([row_count, total_sec], axis=1)

        #average seconds elapsed per user.
        avg_sec = sessions.groupby('user_id')[['secs_elapsed']].mean()

        #adding feature to new_features.
        new_features = pd.concat([new_features, avg_sec], axis=1)

        #rename columns so there isn't any overlap.
        new_features.columns.values[5] = 'total_secs'
        new_features.columns.values[6] = 'avg_secs'

        #frequency of action by user.
        action_freq = pd.DataFrame({'action_count' : sessions.groupby( ["user_id", "action"] ).size()}).reset_index()

        #change action_freq from long to wide.
        action_wide = action_freq.pivot(index = 'user_id', columns = 'action', values = 'action_count')

        #add action type info to new_features data set.
        new_features = pd.concat([new_features, action_wide], axis=1)

        #frequency of action detail by user.
        action_detail_freq = pd.DataFrame({'action_detail_count' : sessions.groupby( ["user_id", "action_detail"] ).size()}).reset_index()

        #change action_detail_freq from long to wide.
        action_detail_wide = action_detail_freq.pivot(index = 'user_id', columns = 'action_detail', values = 'action_detail_count')

        #add action type info to new_features data set
        new_features = pd.concat([new_features, action_detail_wide], axis=1)

        #fill all NAs in new_features with zeros (since all features are frequencies)
        new_features1 = new_features.fillna(0)
        #print(new_features1)
        return new_features1

    def join_data(self,user,session,gender):
        pass

        




