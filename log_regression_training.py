import pandas as pd
import numpy as np
import datetime as dt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

models = [
    LogisticRegression(random_state=0, max_iter=500),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    print("Started processing model family: ", model_name)
    accuracies = cross_val_score(model, os_data_X, np.ravel(os_data_y), scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
        print("Processed: ", model_name, "; fold =", fold_idx, "; accuracy ", accuracy)

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# import seaborn as sns
# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()


agg = cv_df.groupby('model_name').accuracy.mean()

print(agg)

print('We are done. That is all, folks!')
finish_time = dt.datetime.now()
print("Finished at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)
print('We are done. That is all, folks!')
