### First DNN architecture
### Author Tao Sheng 10/19/2018

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K
import sklearn.metrics as skm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Keras Sequential Models
model = Sequential()
model.add(Dense(128, input_dim=10001, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(2, activation='sigmoid'))
model.summary()

### Compilation
# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', f1])



## Run the model
results = list()
sample_size = list()
fprs = list()
tprs = list()
aucs = list()
for k in range(0, 24):
    fx_n = '/mnt/data0/tao/ECG/hdf5/ECG_H' + str(k) + '_Dataset.h5'
    fy_n = '/mnt/data0/tao/ECG/hdf5/ECG_H' + str(k) + '_Label.h5'
    x_n = 'ECG_H' + str(k)    
    with h5py.File(fx_n, 'r') as hf:
        X_H = hf["x_n"][:]
    with h5py.File(fy_n, 'r') as hf:
        Y_H = hf["x_n"][:]
    x_train, x_test, y_train, y_test = train_test_split(X_H, Y_H, test_size=0.1)
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, 
                        validation_data=(x_test, y_test))
    preds = model.predict(x_test, verbose=1)
    gt = np.argmax(y_test, axis=1)
    binary_gt = gt == 0
    binary_probs = preds[..., 0]
    fpr, tpr, _ = skm.roc_curve(binary_gt.ravel(), binary_probs.ravel())
    auc = skm.auc(fpr, tpr)
    results.append([k, max(history.history['acc']), max(history.history['f1']), 
                    max(history.history['val_acc']), max(history.history['val_f1'])])
    sample_size.append([len(x_train), len(x_test)])
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(auc)