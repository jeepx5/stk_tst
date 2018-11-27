import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error




df = pd.read_csv('./dataset/international-airline-passengers.csv',
                      usecols = [1], engine='python', skipfooter = 3)

dataset = df.values
dataset = dataset.astype('float32')
#plt.plot(dataset)
#plt.show()

#normailize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# plt.plot(dataset)
# plt.show()

#split into train and test sets
train_size = int(len(dataset)*0.67)
test_size = int(len(dataset)-train_size)

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

#convert an array of values into a dataset matrix
def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)


look_back =3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#print(trainY, testY)

# reshape input to be[samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


#create and fit LSTM network
model = Sequential()
model.add(LSTM(16, input_shape=(1, look_back)))
model.add(Dense(1))
model.add(Activation( 'relu'))
model.compile(
              loss = 'mean_squared_error',
              optimizer = 'adam',metrics = ['accuracy'])
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
#calculate the rms error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('train score: %.2f RMSE'%(trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('test score: %.2f RMSE'%(trainScore))


#shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back-1:len(trainPredict)+look_back-1, :] = trainPredict
#Shift test prediction for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+look_back*2:len(trainPredict)+look_back*2+len(testPredict), :] = testPredict

plt.plot(scaler.inverse_transform((dataset)))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
