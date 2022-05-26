# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:43:06 2022

@author: kaito
"""

import pandas as pd
import tensorflow as tf
from validation_learn import initialsetting
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import device_lib 

train_path = "input/train.csv"
test_path = "input/test.csv"

X,X_test,Y,test = initialsetting(train_path,test_path)
features = X_test.columns

def my_model():
    activation = 'swish'
    inputs = Input(shape=(len(features)))
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation=activation,
             )(inputs)
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

plot_model(my_model(), show_layer_names=False, show_shapes=True)

def main():
    
    
    
if __name__ == '__main__':
    main()