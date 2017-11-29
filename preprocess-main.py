import pandas as pd
from preprocesser import Preprocesser

### Create proprocessor object
airbnb_preprocess = Preprocesser(target_var='country_destination')



### Clean age_gender_bkts.csv

#age_gender = pd.read_csv("data/age_gender_bkts.csv")
# Delete 'year' column (all values are 2015)
#age_gender.drop(labels='year', axis=1, inplace=True)
#new_gender_features = airbnb_preprocess.transform_gender(age_gender)



### Clean sessions.csv

#Read sessions data and create new features from this data set
log_data = pd.read_csv('data/sessions.csv', encoding='utf-8')
#set user_id field to DataFrame index
log_data.set_index('user_id', inplace = True)
new_log_features = airbnb_preprocess.transform_log(log_data.head(300))


### Transform language

#Read countries data and create new features from this data set
new_language_features = pd.read_csv('data/language_dist.csv', encoding='utf-8')


### Clean User data (test and train sets)

# Read test, training data
train = pd.read_csv('data/train_users_2.csv', encoding='utf-8')
train.set_index('id',inplace=True)

test = pd.read_csv('data/test_users.csv', encoding='utf-8')
test.set_index('id',inplace=True)


# Join language distance data before transforming user data

train = train.reset_index().merge(new_language_features,how='left',on='language').set_index('id')
test = test.reset_index().merge(new_language_features,how='left',on='language').set_index('id')



train_clean = airbnb_preprocess.transform_user(train,missing_data_strategy=None)
test_clean = airbnb_preprocess.transform_user(test,missing_data_strategy=None)


train_full_feature = airbnb_preprocess.join_data(user=train_clean,
                                    gender = None,
                                    session = new_log_features)

test_full_feature = airbnb_preprocess.join_data(user=test_clean,
                                  gender= None,
                                  session=new_log_features)

train_full_feature.to_csv('data/train_with_session_language.csv')
test_full_feature.to_csv('data/test_with_session_language.csv')


# Just preprocess user data. Take no action on missing values
#train_raw = airbnb_preprocess.transform_user(train,missing_data_strategy=None)
#test_raw = airbnb_preprocess.transform_user(test,missing_data_strategy=None)

#train_raw.to_csv("data/raw-user-train.csv")
#test_raw.to_csv('data/raw-user-test.csv')


# User data only. Oversample data with no missing values

#train_impute = airbnb_preprocess.transform_user(train,missing_data_strategy='bag_impute')
#test_impute = airbnb_preprocess.transform_user(test,missing_data_strategy='impute')

#train_impute.to_csv("data/impute-user-train.csv")
#test_impute.to_csv('data/impute-user-test.csv')


