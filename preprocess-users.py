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



