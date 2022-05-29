# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:43:06 2022

@author: kaito
"""

import datetime
import random


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from validation_learn import initialsetting
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Concatenate,BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import device_lib 
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,roc_curve,auc

train_path = "../input/train.csv"
test_path = "../input/test.csv"
X,X_test,Y,test = initialsetting(train_path,test_path)
features = X_test.columns

EPOCHS = 200
EPOCHS_COSINEDECAY = 150
CYCLES = 1
VERBOSE = 0 # set to 0 for less output, or to 2 for more output
DIAGRAMS = True
BATCH_SIZE = 2048
ONLY_FIRST_FOLD = False

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

def my_model():
    activation = 'swish'
    inputs = Input(shape=(len(features)))
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation=activation,
             )(inputs)
    x = BatchNormalization()(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation=activation,
             )(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation=activation,
             )(x)
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation=activation,
             )(x)
    x = Dense(1, #kernel_regularizer=tf.keras.regularizers.l2(1e-6),
              activation='sigmoid',
             )(x)
    model = Model(inputs, x)
    return model

def fit_model(fold,X_train, Y_train, X_val=None, Y_val=None, run=0):
    
    history_list,score_list = [],[]
    
    start_time = datetime.datetime.now()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    if X_val is not None:
        X_val = scaler.transform(X_val)
        validation_data = (X_val, Y_val)
    else:
        validation_data = None

    lr_start=0.01
    
    if X_val is not None: # use early stopping
        epochs = EPOCHS
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
                               patience=4, verbose=VERBOSE)
        es = EarlyStopping(monitor="val_loss",
                           patience=12, 
                           verbose=1,
                           mode="min", 
                           restore_best_weights=True)
        callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]
        
    # Construct and compile the model
    model = my_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_start),
                  metrics='AUC',
                  loss=tf.keras.losses.BinaryCrossentropy())

    # Train the model
    history = model.fit(X_train, Y_train, 
                        validation_data=validation_data, 
                        epochs=epochs,
                        verbose=VERBOSE,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=callbacks)
    
    history_list.append(history.history)
    
    global min_loss,best_fold
    if fold == 0:
        min_loss = history_list[0]['loss'][-1]
        best_fold = fold
    elif min_loss > history_list[0]['loss'][-1]:
        print(f"Update best Fold is {best_fold} to {fold}")
        best_fold = fold    
    else: pass
    
    if X_val is None:
        print(f"Training loss: {history_list[0]['loss'][-1]:.4f}")
    else:
        lastloss = f"Training loss: {history_list[0]['loss'][-1]:.4f} | Val loss: {history_list[0]['val_loss'][-1]:.4f}"
        
        # Inference for validation
        Y_val_pred = model.predict(X_val, batch_size=len(X_val), verbose=0)
        
        # Evaluation: Execution time, loss and AUC
        score = roc_auc_score(Y_val, Y_val_pred)
        print(f"Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}"
              f" | {lastloss} | AUC: {score:.5f}")
        score_list.append(score)
        
        
        fig, axs = plt.subplots(2,3,figsize=(12,8))
        for f, ax in zip(history_list[0], axs.ravel()):
            ax.set_xlabel(f)
            ax.plot(history_list[0][f])
        savepath = "output/learn" + str(fold) + ".png"
        plt.savefig(savepath)
        
        if DIAGRAMS and fold == 0 and run == 0:
            # Plot y_true vs. y_pred
            plt.figure(figsize=(10, 4))
            plt.hist(Y_val_pred[Y_val == 0], bins=np.linspace(0, 1, 21),
                     alpha=0.5, density=True)
            plt.hist(Y_val_pred[Y_val == 1], bins=np.linspace(0, 1, 21),
                     alpha=0.5, density=True)
            plt.xlabel('y_pred')
            plt.ylabel('density')
            plt.title('OOF Predictions')
            plt.savefig("output/OOF Predictions.png")
    print(f"OOF AUC:   {np.mean(score_list):.5f}")
    
    # Inference for test"
    Y_test_pred = model.predict(X_test, batch_size=BATCH_SIZE,verbose=len(X_test))
    
    callbacks, es, lr, history = None, None, None, None
    
    return Y_test_pred


def main():
    result_list = []
    plot_model(my_model(), show_layer_names=False, show_shapes=True)
    
    print(f"{len(features)} features")

    kfold = KFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X,Y)):
        X_train, Y_train = X.iloc[train_idx,:], Y[train_idx]
        X_val, Y_val = X.iloc[val_idx,:], Y[val_idx]
          
        result = fit_model(fold, X_train, Y_train, X_val, Y_val)
        result_list.append(result)
        
        if ONLY_FIRST_FOLD: break # we only need the first fold
        
    test_pred = result_list[best_fold].reshape([len(test.id)])

    output = pd.DataFrame({"id": test.id, "target" : test_pred})
    output.to_csv("output/submission.csv",index=False)
        
if __name__ == '__main__':
    main()