#!/usr/bin/env python
# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import feat_extract
from feat_extract import *
import time
import argparse
import numpy as np
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD

from tensorflow.keras.callbacks import ModelCheckpoint
import os
import os.path as op
from sklearn.model_selection import train_test_split
import tensorflow as tf


def predict(args):
    if op.exists(args.model):
        model = keras.models.load_model(args.model)
        predict_feat_path = 'predict_feat.npy'
        predict_filenames = 'predict_filenames.npy'
        filenames = np.load(predict_filenames)
        X_predict = np.load(predict_feat_path)
        print(X_predict)
        X_predict = np.expand_dims(X_predict, axis=2)

        pred = model.predict_classes(X_predict)
        #pred = model.predict(X_predict)

        for pair in list(zip(filenames, pred)):

            print(pair)

    elif input('Model not found. Train network first? (Y/n)').lower() in ['y', 'yes', '']:
        train()
        predict(args)




def main(args):
    predict(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--train',             action='store_true',                           help='train neural network with extracted features')
    parser.add_argument('-m', '--model',             metavar='path',     default='trained_model3.h5',help='use this model path on train and predict operations')
    #parser.add_argument('-e', '--epochs',            metavar='N',        default=600,              help='epochs to train', type=int)
    parser.add_argument('-p', '--predict',           action='store_true',                           help='predict files in ./predict folder')
    # parser.add_argument('-P', '--real-time-predict', action='store_true',                           help='predict sound in real time')
    #parser.add_argument('-v', '--verbose',           action='store_true',                           help='verbose print')
    #parser.add_argument('-s', '--log-speed',         action='store_true',                           help='performance profiling')
    #parser.add_argument('-b', '--batch-size',        metavar='size',     default=64,                help='batch size', type=int)
    args = parser.parse_args()
    main(args)