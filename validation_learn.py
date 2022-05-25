#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:19:48 2022

@author: arasekaito
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold


train = pd.read_csv("input/train.csv")
test  = pd.read_csv("input/test.csv")
scaler = StandardScaler()

uni = train.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
uni_test = test.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
train['f_27'] = uni
test['f_27'] = uni_test

X = train.drop(["target"],axis=1)

for df in [X,test]:
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
X_test = pd.DataFrame(scaler.transform(test))
k_fold = KFold(n_splits=5, shuffle=True, random_state=1)

params = {"boosting_type": "gbdt",
          "n_estimators": 250,
          "num_leaves": 50,
          "learning_rate":0.1,
          "colsample_bytree": 0.9,
          "subsample": 0.8,
          "reg_alpha": 0.1,
          "objective": "binary",
          "metric": "auc",
          "random_state": 1}

Y = train['target']
model = LGBMClassifier(**params).fit(X,Y,
                                     eval_set=[(X,Y)])

predictions = model.predict(X_test)

output = pd.DataFrame({"id": test.id, "target" : predictions})

output.to_csv("output/submission.csv",index=False)