import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tkinter as tk
from keras.models import load_model
from datetime import datetime
from datetime import date, timedelta
import time
import os.path
from os import path
from generateModel import trainModel
import sys
from tkinter import messagebox
import time
from threading import Thread, Lock

def showMessage(message):
    win = tk.Toplevel()
    win.wm_title("")

    l = tk.Label(win, text=message)
    l.grid(row=0, column=0)

    b = tk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=1, column=0)


#This function takes in a stock name string and a datetime.datetime object to predict prices
def predictPrice(stock, date):
    #Check to see if the model exists already for this stock
    if not path.exists("./models/" + (stock + ".h5")):
        try:
            #Model doesn't already exist. Calling trainModel to make a new one
            mutex = Lock()
            mutex.acquire()

            #showMessage("Model for this stock doesn't exist. Attempting to make a new model. Please Wait...")
            print("Model for this stock doesn't exist. Attempting to make a new model.")
            mutex.release()
            trainModel(stock)
        #Something went wrong with making a new model
        except:
            showMessage("Failed to train new stock model. Verify spelling?")
            print("Failed to train new stock model. Verify spelling?")
            return

    #Read in data from yahoo finance database from the start date until today
    df = web.DataReader(stock, data_source='yahoo', start='2018-01-01', end=datetime.today().strftime('%Y-%m-%d'))

    data = df.filter(['Close'])

    dataset = data.values
    training_data_length = math.ceil(len(dataset) * .8)

    # Scale the data
    # Compute min and max
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Split data into x-train and y_train datasets
    train_data = scaled_data[0:training_data_length, :]
    x_train = []
    y_train = []

    for x in range(90, len(train_data)):
        x_train.append(train_data[x - 90:x, 0])
        y_train.append(train_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = load_model("./models/" + (stock + ".h5"))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    test_data = scaled_data[training_data_length - 90:, :]

    # Create the datasets x_test and y_test
    x_test = []

    for z in range(90, len(test_data)):
        x_test.append(test_data[z - 90:z, 0])

    # Convert to numpy array
    x_test = np.array(x_test)

    # Reshape data for LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict
    predict = model.predict(x_test)
    predict = scaler.inverse_transform(predict)

    valid = data[training_data_length:]
    valid['Predictions'] = predict

    try:
        #Search for a matching date and pull info from the Close section
        result = str(valid.loc[date, :'Close'])
        #For some reason it has extra stuff at the end of the string, call this to remove the excess
        result = result[9:result.find('Name')]
        showMessage("\nThe predicted closing price for " + stock + " on " + date + " is: " + result + "\n")
        print("\nThe predicted closing price for " + stock + " on " + date + " is: " + result + "\n")
    except:
        showMessage("Could not find the date and closing price specified. Model may need to be retrained with current date")
        print("Could not find the date and closing price specified. Model may need to be retrained with current date")


def callPredict(wantedStock, wantedDate):
    try:
        if not datetime.strptime(wantedDate, '%Y-%m-%d') < (datetime.now() - timedelta(91)) or datetime.strptime(wantedDate,'%Y-%m-%d') > datetime.now():
            predictPrice(wantedStock, wantedDate)
        else:
            showMessage("Invalid date, try again")
    except ValueError:
        showMessage("Invalid date, try again")

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        sys.exit()


def main():
    master = tk.Tk()
    master.title("Stock Predictor")
    tk.Label(master, text="Enter a stock to predict the price of: ").grid(row=0)
    tk.Label(master, text="Enter a date between " + str((date.today() - timedelta(90))) + " and today " + str(date.today()) + " to predict the price of (yyyy-mm-dd): ").grid(row=1)

    wantedStockEntry = tk.Entry(master)
    wantedDateEntry = tk.Entry(master)

    wantedStockEntry.grid(row=0, column=1)
    wantedDateEntry.grid(row=1, column=1)
    tk.Button(master, text='Submit', command=lambda: callPredict(wantedStockEntry.get(), wantedDateEntry.get())).grid(row=3, column=1, sticky=tk.W, pady=4)
    tk.Button(master, text='Quit', command=lambda:on_closing()).grid(row=3, column=2, sticky=tk.W, pady=4)

    master.protocol("WM_DELETE_WINDOW", on_closing)
    master.mainloop()


if __name__ == '__main__':
    main()