from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras import backend as K
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
import numpy

from numpy import array
from keras.models import Sequential
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# 导入数据，前8000个正常样本，剩下的样本包括正常和异常时间序列，每个样本是1行48列
dataset = read_csv('randperm_zerone_Dataset.csv')
values = dataset.values
XY = values
n_train_hours1 = 7000
n_train_hours3 = 8000
trainX = XY[:n_train_hours1, :]
validX = XY[n_train_hours1:n_train_hours3, :]
testX = XY[n_train_hours3:, :]
train3DX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
valid3DX = validX.reshape((validX.shape[0], validX.shape[1], 1))
test3DX = testX.reshape((testX.shape[0], testX.shape[1], 1))
# 编码器
sequence = train3DX
# reshape input into [samples, timesteps, features]
n_in = 48
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in, 1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
model.summary()
# fit model
history = model.fit(train3DX, train3DX, shuffle=True, epochs=300, validation_data=(valid3DX, valid3DX))
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='valid')
pyplot.legend()
pyplot.show()
# demonstrate recreation
yhat = model.predict(sequence)
ReconstructedData = yhat.reshape((yhat.shape[0], -1))
numpy.savetxt("ReconstructedData.csv", ReconstructedData, delimiter=',')
