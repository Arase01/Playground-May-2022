# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:16:54 2022

@author: kaito
"""

import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassfier
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier


train = pd.read_csv("input/train.csv")
test  = pd.read_csv("input/test.csv")

uni = train.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
uni_test = test.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
train['f_27'] = uni
test['f_27'] = uni_test

features = train['id']
features_test = test['id']
for i in range(len(train.columns)):
    if train.columns[i] == "id" or train.columns[i] == "target": pass
    else: 
        features = pd.concat([features,train[train.columns[i]]],axis=1)
        
for i in range(len(test.columns)):
    if test.columns[i] == "id" or test.columns[i] == "target": pass
    else: 
        features_test = pd.concat([features_test,test[test.columns[i]]],axis=1)
    
for df in [features,features_test]:
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

model = RandomForestClassifier(n_estimators=100,random_state=1)

Y = train['target']

model.fit(features,Y)
predictions = model.predict(features_test)

output = pd.DataFrame({"id": test.id, "target" : predictions})

output.to_csv("output/submission.csv",index=False)