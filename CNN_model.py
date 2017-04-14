#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import logging
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers import Input, merge, Merge, LSTM, Bidirectional
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from model_input_processing import dataset_generator

if __name__ == '__main__':
    batch_size = 32
    nb_classes = 3
    nb_epoch = 200

    logging.warning('loading data...')

    X_train, y_train, X_test, y_test = dataset_generator('lynch.csv', 'vectors.txt', 0.8, 'xinsong')
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

#    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

#    input_size = 

#    print(input_size)
    print(len(Y_train))
    print(len(Y_test))

#    X_train = X_train.tolist()
#    X_test = X_test.tolist()

    logging.warning('building model')


    model = Sequential()
    model.add(Conv1D(filters = 32, kernel_size = 2, strides = 1, padding = 'valid', input_shape = (29, 50)))
#    model.add(Conv1D(filters = 32, kernel_size = 2, strides = 1, padding = 'valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters = 64, kernel_size = 2, strides = 1, padding = 'same'))
    model.add(Conv1D(filters = 128, kernel_size = 2, strides = 1, padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

    logging.warning('training...')

    #training the model with checkpointer
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5",  monitor='val_acc',verbose=2, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=50, epochs=3000, verbose=2,  validation_data=(X_test, Y_test), callbacks=[checkpointer])

    score = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)

