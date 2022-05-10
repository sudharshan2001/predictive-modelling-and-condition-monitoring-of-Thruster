import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Input
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model

from sklearn.preprocessing import MinMaxScaler, StandardScaler

File_name = 'SN01_20_30.csv'
df=pd.read_csv(File_name)

df_thrust_ts = df.loc[:, ["Unnamed: 0","thrust"]]
df_thrust_ts.index = df_thrust_ts["Unnamed: 0"]
ts = df_thrust_ts.drop("Unnamed: 0",axis=1)

df['Date'] = pd.to_datetime(df['Unnamed: 0'])

# Last 1 minute of data will be for testing
train, test = df.loc[df['Date'] <= '2021-03-04 10:15:00.000'], df.loc[df['Date'] > '2021-03-04 10:15:00.010']

scaler = StandardScaler()
scaler = scaler.fit(train[['thrust', 'mfr']])

train[['thrust', 'mfr']] = scaler.transform(train[['thrust', 'mfr']])
test[['thrust', 'mfr']] = scaler.transform(test[['thrust', 'mfr']])

train.drop(['Unnamed: 0', 'ton'], axis=1, inplace=True)

seq_size = 70  # you can increase the seq_size to get better result


def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):

        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

trainX, trainY = to_sequences(train[['thrust', 'mfr']], train[['thrust', 'mfr']], seq_size)
testX, testY = to_sequences(test[['thrust', 'mfr']], test[['thrust', 'mfr']], seq_size)

# Here we use LSTM autoencoder  to compress data using an encoder and decode it to retain original structure using a decoder.
# we use the Latent Representation of data from Autoencoders, from which we can detect the outliers as it reduces the dimension
lr= 0.01


model = Sequential()
model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(RepeatVector(trainX.shape[1]))

model.add(LSTM(64, return_sequences=True))

model.add(TimeDistributed(Dense(trainX.shape[2])))

adam_ = tf.keras.optimizers.Adam(learning_rate=lr)

model.compile(optimizer=adam_, loss='mse',metrics="acc")
model.summary()

history = model.fit(trainX, trainX, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

#Saving the model
model.save('model_file/lstm_AE.h5')

#Plotting the performance
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()


