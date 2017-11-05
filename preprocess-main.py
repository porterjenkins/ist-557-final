import pandas as pd
from preprocesser import Preprocesser

### Create proprocessor object
airbnb_preprocess = Preprocesser()

### Clean age_gender_bkts.csv

age_gender = pd.read_csv("data/age_gender_bkts.csv")
# Delete 'year' column (all values are 2015)
age_gender.drop(labels='year', axis=1, inplace=True)
new_gender_features = airbnb_preprocess.transform_gender(age_gender)



### Clean sessions.csv

#Read sessions data and create new features from this data set
log_data = pd.read_csv('data/sessions.csv', encoding='utf-8')
#set user_id field to DataFrame index
log_data.set_index('user_id', inplace = True)
new_log_features = airbnb_preprocess.transform_log(log_data)

### Clean User data (test and train sets)

# Read test, training data
train = pd.read_csv('data/train_users_2.csv', encoding='utf-8')
train.set_index('id',inplace=True)

test = pd.read_csv('data/test_users.csv', encoding='utf-8')
test.set_index('id',inplace=True)


train_clean = airbnb_preprocess.transform_user(train,bagging=True)
test_clean = airbnb_preprocess.transform_user(test,bagging=False)







