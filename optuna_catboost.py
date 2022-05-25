# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:36:37 2022

@author: kaito
"""

import pandas as pd
import matplotlib.pyplot as plt
import optuna
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier


def objective(X,Y,trial):
    params = {
        'iterations' : trial.suggest_int('iterations', 100,5000),                         
        'depth' : trial.suggest_int('depth', 4, 10),                                       
        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),               
        'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait' :trial.suggest_int('od_wait', 10, 50),
        'task_type': "GPU"
        }
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    
    model = CatBoostClassifier(**params)
    model.fit(X_train,Y_train)
    
    score = model.score(X_test,Y_test)
    return 1 - score
    
def main(): 
    train = pd.read_csv("input/train.csv")
    scaler = StandardScaler()
    
    uni = train.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
    train['f_27'] = uni
    
    X = train.drop(["target"],axis=1)
    
    for df in [X]:
        df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
        df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
        i_00_01_26 = df.f_00 + df.f_01 + df.f_26
        df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
    X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
    Y = train['target']
    
    Objective = partial(objective,X,Y)
    study = optuna.create_study()
    study.optimize(Objective, n_trials=100)
    print(study.best_trial)

if __name__ == '__main__':
    main()
