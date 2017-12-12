## Note: This is Python 3.x file

import pandas as pd
from preprocesser import Preprocesser

### Create proprocessor object
airbnb_preprocess = Preprocesser(target_var='country_destination')


### Clean sessions.csv

#Read sessions data and create new features from this data set
log_data = pd.read_csv('data/sessions.csv', encoding='utf-8')

#set user_id field to DataFrame index
log_data.set_index('user_id', inplace = True)
new_log_features = airbnb_preprocess.transform_log(log_data)

### Transform language

#Read countries data and create new features from this data set
new_language_features = pd.read_csv('data/language_dist.csv', encoding='utf-8')


### Clean User data (test and train sets)

# Read test, training data
train = pd.read_csv('data/train_users_2.csv', encoding='utf-8')
train.set_index('id',inplace=True)

test = pd.read_csv('data/test_users.csv', encoding='utf-8')
test.set_index('id',inplace=True)





train_clean = airbnb_preprocess.transform_user(train,missing_data_strategy=None)
test_clean = airbnb_preprocess.transform_user(test,missing_data_strategy=None)


train_full_feature = airbnb_preprocess.join_data(user=train_clean,
                                    gender = None,
                                    session = new_log_features)

test_full_feature = airbnb_preprocess.join_data(user=test_clean,
                                  gender= None,
                                  session=new_log_features)

train_full_feature.to_csv('data/train_with_session_language_fill_all_nan.csv')
test_full_feature.to_csv('data/test_with_session_language_fill_all_nan.csv')




