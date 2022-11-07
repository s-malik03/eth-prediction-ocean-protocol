# -*- coding: utf-8 -*-
"""
Created on 2022

@author:Jason
"""
import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


previous_date = 72
future_date_to_predict = 15
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predict", action="store_true", help="Predict model")
    args = parser.parse_args()
    return args

def oneHot(y, num_class=3):
    y_out = np.zeros(num_class)
    y_out[y] = 1
    return y_out
def readTrain(file):
    train = pd.read_csv(file)
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
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    close_avg = np.mean(train)['close']
    close_dis = np.max(train)['close'] - np.min(train)['close']
    print (close_avg, close_dis)
    print (np.max(train), np.min(train))
    parameters = {"mean": np.mean(train), "dis": np.max(train) - np.min(train)}
    print (parameters)
    return train_norm, parameters

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
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, input_length=shape[1], input_dim=shape[2], dropout=0.0, recurrent_dropout=0.0,kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.5))
#    lstm_model.add(Dense(5,activation='tanh'))
    lstm_model.add(Dense(64))
    lstm_model.add(Dense(15))
    lstm_model.compile(loss='mean_squared_error', optimizer="adam")
    return lstm_model

def output_loss_figure(history, time):
    # summarize history for loss 
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.savefig('loss_' + time + '.png')
    
def output_prediction_result(x_pred, x_true, current_time=""):
    plt.clf()
    plt.plot(x_true)
    plt.plot(x_pred)
    plt.title('Eth price prediction from 2018/05/15')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(['Real Price', 'Prediction'], loc='upper left') 
    plt.savefig('result_' + current_time + '.png')
    
def calc_nmse(y, yhat) -> float:
    assert len(y) == len(yhat)
    mse_xy = np.sum(np.square(np.asarray(y) - np.asarray(yhat)))
    mse_x = np.sum(np.square(np.asarray(y)))
    nmse = mse_xy / mse_x
    return nmse

def train_stock_prediction_model(file_name, output=False):
    train = readTrain(file_name)

    # Augment the features (year, month, date, day)
    train_Aug = augFeatures(train)

    # Normalization
    train_norm, parameters = normalize(train_Aug)
    with open("parameters.pkl","wb") as f:
        pickle.dump(parameters, f)
    print (train_norm.head())
    X_train, Y_train = buildTrain(train_norm, previous_date, future_date_to_predict)
    
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
        output_loss_figure(history, current_time)
        stock_prediction(X, Y, model_file_name, parameters)
    
def stock_prediction(X, Y, model_file_name, parameters, current_time=""):
    close_avg = parameters['mean']['close']
    close_dis = parameters['dis']['close']
    model = load_model(model_file_name)
    x_pred = model.predict(X)*close_dis+ close_avg
    x_true = Y*close_dis+close_avg

    output_prediction_result(x_pred, x_true, current_time=current_time)
    
def predict_from_ocean_dataset(df, model_file_name, parameters):
    df["Date"] = pd.to_datetime(df["ds"])
    df["close"] = df["y"]
    df["day"] = df["Date"].dt.dayofweek
    df["hour"] = df["Date"].dt.hour
   
    df = df.drop(["Date"], axis=1)
    df = df.drop(["y"], axis=1)
    df = df.drop(["ds"], axis=1)
    df = df.iloc[: , 1:]
    data = np.array([df.iloc[-future_date_to_predict:,:]["close"]])
    
    df["close"] = (df["close"] - parameters["mean"]["close"]) / parameters["dis"]["close"]
    df["day"] = (df["day"] - parameters["mean"]["day"]) / parameters["dis"]["day"]
    df["hour"] = (df["hour"] - parameters["mean"]["hour"]) / parameters["dis"]["hour"]
    
    #print (df)
    #df = df.apply(lambda x: (x - close_avg) / (np.max(x) - close_dis))
    close_avg = parameters["mean"]["close"]
    close_dis = parameters["dis"]["close"]

    #test = np.array([df.iloc[-future_date_to_predict-previous_date:-future_date_to_predict,:]])
    test = np.array([df.iloc[-previous_date:,:]])
    print (test)
    print (test.shape)
    model = buildManyToOneModel(test.shape)
    model.load_weights(model_file_name)
    #model = load_model(model_file_name)
    model.summary()
    pred_vals = model.predict(test)*close_dis+close_avg
    #print (data)
    print (pred_vals)
    #print ("NMSE:" + str(calc_nmse(data, pred_vals)))
    
    plt.clf()
    plt.plot(pred_vals[0])
    #plt.plot(data[0])
    plt.title('Eth price prediction for future 12 hours')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(['Prediction'], loc='upper left') 
    plt.savefig('result_future12.png')
    
    return pred_vals

if __name__ == '__main__':
    
    args = parse_args()

    if args.predict == False:
        train_stock_prediction_model(file_name='./dataset/EHTUSDhrDataset.csv', output=True)
    else:
        with open("parameters.pkl","rb") as f:
            parameters = pickle.load(f)
        df = pd.read_csv('./dataset/ETHUSD_1106.csv')
        #pred_vals = predict_from_ocean_dataset(df, './model/eth_model_00-57-46.h5', parameters)
        pred_vals = predict_from_ocean_dataset(df, './model/eth_model_19-35-17.h5', parameters)
    
