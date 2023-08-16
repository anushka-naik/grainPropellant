from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from itertools import product
import math
from math import pi
import pandas as pd
import time
import torch
from flask_socketio import SocketIO, emit
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from tensorflow.keras.callbacks import Callback

# Data Preparation
class CustomCheckpoint(Callback):
    def __init__(self, filepath, save_freq):
        super().__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.prev_checkpoint = None
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            current_checkpoint = self.filepath.format(epoch=epoch+1)
            self.model.save(current_checkpoint)
            
            if self.prev_checkpoint and os.path.exists(self.prev_checkpoint):
                os.remove(self.prev_checkpoint)
            self.prev_checkpoint = current_checkpoint



def gru_model():
    df = pd.read_csv('dataset.csv')
    df = df[df['BurnRate'].str.len() > 2]

    df['BurnRate'] = df['BurnRate'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    max_length = df['BurnRate'].apply(len).max()
    df['BurnRate'] = df['BurnRate'].apply(lambda x: x + [0] * (max_length - len(x)))

    X = df.drop(columns='BurnRate').values
    Y = np.stack(df['BurnRate'].values)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scaling the data
    scaler_X = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Model Building

    model = Sequential()
    model.add(GRU(50, input_shape=(X_train_scaled.shape[1], 1), return_sequences=True))
    model.add(GRU(50))
    model.add(Dense(max_length))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Reshaping X data to fit LSTM 3D input (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

   # Create the directories if they don't exist
    if not os.path.exists('./models/gru/checkpoints/'):
        os.makedirs('./models/gru/checkpoints/')

    checkpoint_callback = CustomCheckpoint(filepath="./models/gru/checkpoints/gru_model_{epoch:02d}.h5", save_freq=5)


    # Training the model with the ModelCheckpoint callback
    model.fit(X_train_reshaped, Y_train, epochs=11, batch_size=32, validation_data=(X_test_reshaped, Y_test), callbacks=[checkpoint_callback])

    model.save('./models/gru/gru_model_final.h5')
    
    return 
    
    

    
    
