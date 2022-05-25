# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:16:54 2022

@author: kaito
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

scaler = StandardScaler()
train = pd.read_csv("input/train.csv")
test  = pd.read_csv("input/test.csv")

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

params={'iterations': 4516, 
        'depth': 10,
        'learning_rate': 0.03946057958646794, 
        'random_strength': 86,
        'bagging_temperature': 0.7680045063918526,
        'od_type': 'Iter', 
        'od_wait': 43,
        'task_type': "GPU"}

X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
X_test = pd.DataFrame(scaler.transform(test),columns=X_test.columns)
Y = train['target']

model = CatBoostClassifier(**params)
model.fit(X,Y)

predictions = model.predict(X_test)

output = pd.DataFrame({"id": test.id, "target" : predictions})

output.to_csv("output/submission.csv",index=False)