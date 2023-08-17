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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

def delete_old_models(directory, extension=".pkl"):
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    for f in files:
        os.remove(os.path.join(directory, f))

def gru_model(num_epochs, batch_size, save_freq):
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

    if not os.path.exists("./models/gru/metrics/"):
        os.makedirs('./models/gru/metrics/')
        
    checkpoint_callback = CustomCheckpoint(filepath="./models/gru/checkpoints/gru_model_{epoch:02d}.h5", save_freq=save_freq)


    # Training the model with the ModelCheckpoint callback
    history = model.fit(X_train_reshaped, Y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test_reshaped, Y_test), callbacks=[checkpoint_callback])

     # Save model history to a CSV file
    history_df = pd.DataFrame(history.history)
    history_csv_path = './models/gru/metrics/history.csv'
    history_df.to_csv(history_csv_path, index=False)

    # Save loss and val_loss plots
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = './models/gru/metrics/loss_plot.png'
    plt.savefig(loss_plot_path)
    plt.close()

    # Move loss plot and metrics to the metrics folder
    if not os.path.exists('./models/gru/metrics/'):
        os.makedirs('./models/gru/metrics/')
    os.rename(loss_plot_path, './models/gru/metrics/loss_plot.png')
    os.rename(history_csv_path, './models/gru/metrics/history.csv')
    
    
    model.save('./models/gru/gru_model_final.h5')
    
    # Taking X_test[0] as a sample input and reshaping it to fit the 3D input shape (samples, timesteps, features)
    sample_input = X_test_reshaped[0].reshape(1, X_test_reshaped.shape[1], 1)

    # Making predictions from the model
    sample_output = model.predict(sample_input)

    # Printing the predicted output
    print(sample_output)
    return 
 
    
def lstm_model(num_epochs, batch_size, save_freq):
    df = pd.read_csv('dataset.csv')
    df = df[df['BurnRate'].str.len() > 2] #ignoring all BurnRates with empty lists

    # Converting the BurnRate string representation of lists into actual lists
    df['BurnRate'] = df['BurnRate'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])

    # Ensure all lists have the same length, if not, pad with zeros (or other appropriate value)
    max_length = df['BurnRate'].apply(len).max()
    df['BurnRate'] = df['BurnRate'].apply(lambda x: x + [0] * (max_length - len(x)))

    ## Prepare the data for LSTM
    X = df.drop(columns='BurnRate').values
    Y = np.stack(df['BurnRate'].values)

    # Reshape X
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Splitting the dataset into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Model Building
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dense(max_length)) # modified Dense layer

    model.compile(optimizer='adam', loss='mean_squared_error')
    
     # Create the directories if they don't exist
    if not os.path.exists('./models/lstm/checkpoints/'):
        os.makedirs('./models/lstm/checkpoints/')
        
    if not os.path.exists("./models/lstm/metrics/"):
        os.makedirs('./models/lstm/metrics/')
    
    checkpoint_callback = CustomCheckpoint(filepath="./models/lstm/checkpoints/lstm_model_{epoch:02d}.h5", save_freq=save_freq)

    # Training
    history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=[checkpoint_callback])

    # Save model history to a CSV file
    history_df = pd.DataFrame(history.history)
    history_csv_path = './models/lstm/metrics/history.csv'
    history_df.to_csv(history_csv_path, index=False)

    # Save loss and val_loss plots
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = './models/lstm/metrics/loss_plot.png'
    plt.savefig(loss_plot_path)
    plt.close()

    # Move loss plot and metrics to the metrics folder
    if not os.path.exists('./models/lstm/metrics/'):
        os.makedirs('./models/lstm/metrics/')
    os.rename(loss_plot_path, './models/lstm/metrics/loss_plot.png')
    os.rename(history_csv_path, './models/lstm/metrics/history.csv')
    
    model.save('./models/lstm/lstm_model_final.h5')

    # Prediction
    predictions = model.predict(X_test[0].reshape(1, X_test.shape[1], X_test.shape[2]))
    
    print("PREDICTIONS")
    print(predictions)
    
    return

        

def svr_model(num_epochs, batch_size, save_freq):
    # 1. Data Preparation
    df = pd.read_csv('dataset.csv')
    df = df[df['BurnRate'].str.len() > 2]
    df['BurnRate'] = df['BurnRate'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    max_length = df['BurnRate'].apply(len).max()
    df['BurnRate'] = df['BurnRate'].apply(lambda x: x + [0] * (max_length - len(x)))

    X = df.drop(columns='BurnRate').values
    Y = np.stack(df['BurnRate'].values)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 2. Model Building
    scaler_X = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_Y = MinMaxScaler().fit(Y_train)
    Y_train_scaled = scaler_Y.transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)

    model = MultiOutputRegressor(SVR(C=1.0, epsilon=0.1, kernel='rbf'))

    # Create the directories if they don't exist
    if not os.path.exists('./models/svr/checkpoints/'):
        os.makedirs('./models/svr/checkpoints/')

    # Save model at intervals
    for epoch in range(num_epochs):
        model.fit(X_train_scaled, Y_train_scaled)
        
        if epoch % save_freq == 0:
            delete_old_models('./models/svr/checkpoints/')
            joblib.dump(model, f'./models/svr/checkpoints/svr_model_epoch_{epoch}.pkl')         
        
    # 3. Training & Evaluation
    predictions = model.predict(X_test_scaled)
    predictions = scaler_Y.inverse_transform(predictions)
    
    mse = mean_squared_error(Y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Save final model
    joblib.dump(model, './models/svr/svr_model_final.pkl')

    return predictions, mse





