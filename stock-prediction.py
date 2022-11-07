# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:50:32 2019

@author:Jason
"""

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from datetime import datetime
selected_model = 0 # 0 for close prediction, 1 for label prediction
class_num = 3
def oneHot(y, num_class=3):
    y_out = np.zeros(num_class)
    y_out[y] = 1
    return y_out
def readTrain(file):
    train = pd.read_csv(file, skiprows=1)
    return train
def augFeatures(train):
    train["Date"] = pd.to_datetime(train["date"])
    #train["year"] = train["Date"].dt.year
    #train["month"] = train["Date"].dt.month
    #train["date"] = train["Date"].dt.day
    train["day"] = train["Date"].dt.dayofweek
    train["hour"] = train["Date"].dt.hour
    return train
def normalize(train):
    global scl
    train = train.drop(["date"], axis=1)
    train = train.drop(["Date"], axis=1)
    train = train.drop(["symbol"], axis=1)
    train = train.drop(["unix"], axis=1)
    train = train.drop(["Volume ETH"], axis=1)
    train = train.drop(["Volume USD"], axis=1)
    train = train.drop(["open"], axis=1)
    train = train.drop(["high"], axis=1)
    train = train.drop(["low"], axis=1)
    
    print(np.mean(train))
    #train_norm = scl.fit_transform(train)
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    close_avg = np.mean(train)['close']
    close_dis = np.max(train)['close'] - np.min(train)['close']
    print (close_avg, close_dis)
    return train_norm, close_avg, close_dis

def buildTrain(train, pastDay=48, futureDay=12):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["close"]))
    return np.array(X_train), np.array(Y_train)

def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]
def splitData(X,Y,rate):
    X_train = X[:int(X.shape[0]*(1-rate))]
    Y_train = Y[:int(Y.shape[0]*(1-rate))]
    X_val = X[int(Y.shape[0]*(1-rate)):]
    Y_val = Y[int(Y.shape[0]*(1-rate)):]
    return X_train, Y_train, X_val, Y_val

def buildManyToOneModel(shape):
    '''
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
  # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model
    '''    
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, input_length=shape[1], input_dim=shape[2], dropout=0.0, recurrent_dropout=0.0,kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.5))
#    lstm_model.add(Dense(5,activation='tanh'))
    lstm_model.add(Dense(64))
    lstm_model.add(Dense(16))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer="adam")
    return lstm_model
def output_result(history, time):
    '''
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()
    '''
    # summarize history for loss 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.savefig('loss_' + time + '.png')

def train_stock_prediction_model(file_name, output=False):
    train = readTrain(file_name)

    # Augment the features (year, month, date, day)
    train_Aug = augFeatures(train)

    # Normalization
    train_norm, close_avg, close_dis = normalize(train_Aug)
    print (train_norm.head())
    X_train, Y_train = buildTrain(train_norm, 60, 1)
    
    X = X_train
    Y = Y_train
    # shuffle the data, and random seed is 10
    X_train, Y_train = shuffle(X_train, Y_train)

    # split training data and validation data
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

    print (X_train.shape, Y_train.shape)
    model = buildManyToOneModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")

    checkpoint_filepath = './tmp/model_checkpoint_{epoch}'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_freq= 2)
    
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback, model_checkpoint_callback])
    
    model_file_name = 'model/eth_model_' + str(current_time) + '.h5'
    model.save(model_file_name)
    if output:
        output_result(history, current_time)
        stock_prediction(X, Y, model_file_name, close_avg, close_dis)
    
def stock_prediction(X, Y, model_file_name, close_avg, close_dis, current_time):
    model = load_model(model_file_name)
    plt.plot(model.predict(X)*close_dis+close_avg)
    plt.plot(Y*close_dis+close_avg)
    plt.savefig('result_' + current_time + '.png')
    
if __name__ == '__main__':
    train_stock_prediction_model(file_name='Bitstamp_ETHUSD_1h.csv', output=True)
    
'''
source = ['Bitstamp_ETHUSD_1h']
for i in range (len(source)):
    train = readTrain(source[i] + ".csv")

# Augment the features (year, month, date, day)
    train_Aug = augFeatures(train)

# Normalization
    train_norm = normalize(train_Aug)
    print (train_norm.head())
    if selected_model == 1:
        if i == 0:
            X_train, Y_train = buildLabelTrain(train_norm, 60, 1)
    # build Data, use last 30 days to predict next 5 days
        else:
            X_train, Y_train = buildMultiLabelTrain(X_train, Y_train, train_norm, 60, 1)
    else:
        if i == 0:
            X_train, Y_train = buildTrain(train_norm, 60, 1)
    # build Data, use last 30 days to predict next 5 days
        else:
            X_train, Y_train = buildMultiTrain(X_train, Y_train, train_norm, 60, 1)

X = X_train
Y = Y_train
# shuffle the data, and random seed is 10
X_train, Y_train = shuffle(X_train, Y_train)

# split training data and validation data
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

print (X_train.shape, Y_train.shape)

if selected_model == 1:
    model = buildManyToOneLabelModel(X_train.shape)
else:
    model = buildManyToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")


checkpoint_filepath = './tmp/model_checkpoint_{epoch}'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq= 2)
now = datetime.now()
current_time = now.strftime("%H-%M-%S")
history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback, model_checkpoint_callback])
model.save('model/eth_model_' + str(current_time) + '.h5')
output_result(history, current_time)



#plt.plot(X_train["Adj Close"])
#plt.plot(model.predict(X))
#plt.plot(Y)
#plt.savefig('result.png')
#plt.plot(Y)
'''