import pandas as pd
from preprocesser import Preprocesser

# Read test, training data
train = pd.read_csv('data/train_users_2.csv', encoding='utf-8')
# set ID field to DataFrame index
train.set_index('id',inplace=True)


test = pd.read_csv('data/test_users.csv', encoding='utf-8')
test.set_index('id',inplace=True)


train_preprocesser = Preprocesser(bagging=True)
train_clean = train_preprocesser.transform(train)

test_preprocesser = Preprocesser(bagging=False)
test_clean = test_preprocesser.transform(test)

#Read sessions data and create new features from this data set
log_data = pd.read_csv('data/sessions.csv', encoding='utf-8')

#set user_id field to DataFrame index
log_data.set_index('user_id', inplace = True)

new_features = Preprocesser.transform_log(log_data)



