import pandas as pd
from pandas import DataFrame


train = pd.read_csv("data/train_users_2.csv")
test = pd.read_csv("data/test_users.csv")

users = test["id"]
n_user_train = train.shape[0]

class_distribution = train.country_destination.value_counts() / float(n_user_train)

top_five_class = list(class_distribution.head().index)

user_list = []
predictions = []

for i in users:
    class_count = 0
    for j in range(5):
        user_list.append(i)
        predictions.append(top_five_class[j])




submission = DataFrame(data=predictions,index=user_list,columns=['country'])
submission.index.rename('id',inplace=True)

submission.to_csv("output/predictions/naive_baseline.csv")

