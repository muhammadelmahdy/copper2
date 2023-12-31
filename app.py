import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError


def predict(arr,model):
  #adjust the date
  arr=arr[:-2] #we discard the Vol. and change%
  timestamp = datetime(2022, 1, 3, 0, 0, 0) #The minimum in the dataframe
  arr[0]=datetime.strptime(arr[0],"%m/%d/%Y")
  arr[0] = (arr[0] - timestamp).total_seconds() / (24 * 60 * 60)
  arr=[arr]
  scaler=StandardScaler()
  arr=scaler.fit_transform(arr)
  arr_new=np.reshape(np.array(arr),(1,1,4))
  preds=model.predict(arr_new)
  return preds
 



def main():
  # Load the pre-trained TensorFlow model
  
  model2 = Sequential()
  model2.add(LSTM(64, input_shape=(1,4)))
  model2.add(tf.keras.layers.Reshape((1,64,), input_shape=(None,64)))
  model2.add(LSTM(32, input_shape=(1,4)))
  model2.add(Dropout(0.2))
  model2.add(Dense(4))
  model2.add(Dropout(0.2))
  model2.add(Dense(1))
  model2.compile(loss = 'mean_squared_error', optimizer = 'Adam',metrics=[RootMeanSquaredError()])
  model2.load_weights('weights.h5')


  st.title('Stock Price Prediction App')

  st.write("Please enter the following data:")

  date = st.date_input("Date")
  open_price = st.number_input("Open Price")
  high_price = st.number_input("High Price")
  low_price = st.number_input("Low Price")

  arr=np.array([date,open_price,high_price,low_price])
  preds=predict(arr,model2)


  st.write("Predicted Close Price:", preds[0][0])


if __name__ == "__main__":
    main()


