import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime
import os

#This function takes in a string of a stock name (ex: TSLA), trains a model with the read data, then saves the model as a .h5 file format
def trainModel(stock):
    #Read in data from yahoo finance database from the start date until today
    df = web.DataReader(stock, data_source='yahoo', start='2018-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    data = df.filter(['Close'])
    dataset = data.values
    training_data_length = math.ceil(len(dataset) * .8)

    training_data_length

    # Scale the data
    # Compute min and max
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Split data into x-train and y_train datasets
    train_data = scaled_data[0:training_data_length, :]
    x_train = []
    y_train = []

    #Set the last 90 days as training data
    for x in range(90, len(train_data)):
        x_train.append(train_data[x - 90:x, 0])
        y_train.append(train_data[x, 0])

    # Convert training sets to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape datasets, LSTM expects 3 dimensions
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Expected  output is (413, 90, 1)

    # LSTM model
    model = Sequential()
    model.add(LSTM(60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(60, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(20))
    model.add(Dense(1))
    # Compile Model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train model
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    model.save("./models/" + (stock + ".h5"))

def main():
    stock = input("Enter stock name to train: ")
    trainModel(stock)

if __name__ == '__main__':
    main()