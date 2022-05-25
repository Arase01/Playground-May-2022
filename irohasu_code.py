import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

scaler = MinMaxScaler()

def preprocess(X):
    count_list = []
    for s in X['f_27']:
        count_list.append(len(set(s)))
    X['f_27'] = count_list
    X['i_02_21'] = (X.f_21 + X.f_02 > 5.2).astype(int) - (X.f_21 + X.f_02 < -5.3).astype(int)
    X['i_05_22'] = (X.f_22 + X.f_05 > 5.1).astype(int) - (X.f_22 + X.f_05 < -5.4).astype(int)
    i_00_01_26 = X.f_00 + X.f_01 + X.f_26
    X['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    X = scaler.fit_transform(X.values)
    return X

features = [col for col in train_df.columns if col not in ['id', 'target']]
X = train_df[features]
X_test = test_df[features]
y = pd.DataFrame(train_df['target'], columns=['target'])

X = preprocess(X)
X_test = preprocess(X_test)
y = y.values

predictions, scores = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(10*"=", f"Fold={fold+1}", 10*"=")

#     X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#     X_valid , y_valid = X.iloc[valid_idx] , y.iloc[valid_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid , y_valid = X[valid_idx] , y[valid_idx]
    
    model = RandomForestClassifier(max_depth=5, random_state=1)
    model.fit(X_train, y_train)
    
    preds_valid = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds_valid)
    print(f'Accuracy score: {acc:5f}\n')
    scores.append(acc)
    test_preds = model.predict(X_test)
    predictions.append(test_preds)
    
score = np.array(scores).mean()
print(f'Mean accuracy score: {score:6f}')
y_pred = mode(predictions).mode[0]
submission = pd.DataFrame({"id": test_df.id, "target" : y_pred})
submission.to_csv('output/submission.csv', index=False)