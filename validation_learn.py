"""
Created on Tue May 24 15:19:48 2022

@author: arasekaito
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.model_selection import KFold

train_path = "input/train.csv"
test_path = "input/test.csv"


def initialsetting(train_path,test_path):
    
    scaler = StandardScaler()
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    
    uni = train.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
    uni_test = test.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
    train['f_27'] = uni
    test['f_27'] = uni_test
    
    X = train.drop(["target"],axis=1)
    X_test = test
    
    for df in [X,X_test]:
        df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
        df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
        i_00_01_26 = df.f_00 + df.f_01 + df.f_26
        df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
    X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(test))
    Y = train['target']
    
    return X,X_test,Y

def main():
    X,X_test,Y = initialsetting(train_path,test_path)
    
    params={'iterations': 284,
        'depth': 10,
        'learning_rate': 0.24641023327616474,
        'random_strength': 0, 
        'bagging_temperature': 0.21482075013237478,
        'od_type': 'Iter', 
        'od_wait': 31,
        'task_type': "GPU"}
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    model = CatBoostClassifier(**params)
    predictions = model.predict(X_test)   
    output = pd.DataFrame({"id": X_test.id, "target" : predictions})
    output.to_csv("output/submission.csv",index=False)

if __name__ == '__main__':
    main()
