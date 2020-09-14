import pandas as pd
import numpy as np
import datetime as dt

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score

from imblearn.over_sampling import SMOTE

start_time = dt.datetime.now()
print("Started at ", start_time)
# read finally preprocessed training set

final_training_df_cached_file = 'interim_data/final_training.csv'
final_training_df = pd.read_csv(final_training_df_cached_file)

# get the target
Y = final_training_df['target']
# get the set of predictors
X = final_training_df.drop(columns=['target'])

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Scaling using the Standard Scaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = scaler.transform(X_test)

os = SMOTE(random_state=0)

columns = X_train.columns
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['target'])

# Converting the dataset in proper LGB format
d_train = lgb.Dataset(os_data_X, label=os_data_y)

# setting up the parameters
number_of_classes = 264

params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_leaves': 200,
    'min_data_in_leaf': 70,
    # 'feature_fraction': 0.8,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 10,
    # 'max_depth': 10,
    'num_class': number_of_classes
}

# training the model
clf = lgb.train(params,
                d_train,
                100,
                verbose_eval=1)  # training the model on 100 epocs

# prediction on the test dataset
y_pred_1 = clf.predict(X_test)
# printing the predictions
print(y_pred_1)

# argmax() method
y_pred_1 = [np.argmax(line) for line in y_pred_1]

# using precision score for error metrics
precision = precision_score(y_pred_1, y_test, average=None).mean()

print(precision)
print('We are done. That is all, folks!')
finish_time = dt.datetime.now()
print("Finished at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)
print('We are done. That is all, folks!')
