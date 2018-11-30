import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#pro = ts.pro_api()
#df = pro.daily(ts_code='600527.SH', start_date = '20160101', end_date='20181112')
#df.to_csv('600527.csv')
df = pd.read_csv('600527.csv')

print(df.keys())

df['date'] = pd.to_datetime(df['trade_date'],format='%Y%m%d', errors='ignore')
#print(df['date'])
#plt.plot(df['date'], df['close'])
#plt.show()



dataset = df['close'].values
dataset = dataset[-1:0:-1]
#print(dataset)


#scaler = MinMaxScaler(feature_range=(0,1))
#dataset = scaler.fit_transform(dataset)

#print(dataset)


#split into train and test sets
train_size = int(len(dataset)*0.67)
test_size = int(len(dataset)-train_size)
print(train_size, test_size, len(dataset))

train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
print(len(train), len(test))

#convert an array of values into a dataset matrix
def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)


look_back =7

#ADF check





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
model.fit(trainX, trainY, epochs=300, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# #invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
#calculate the rms error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('train score: %.2f RMSE'%(trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('test score: %.2f RMSE'%(trainScore))

print(trainPredict.shape)
print(testPredict.shape)
#shift train predictions for plotting

trainPredict=pd.DataFrame(trainPredict.reshape(len(trainPredict),1))
testPredict=pd.DataFrame(testPredict.reshape(len(testPredict),1))

nalst = pd.DataFrame(np.zeros(look_back))
trainPredictPlot = pd.DataFrame(np.empty_like(dataset))
#testPredictPlot = pd.DataFrame(np.empty_like(dataset))

trainPredictPlot.iloc[look_back:look_back+len(trainPredict),:]=trainPredict
#testPredictPlot.iloc[look_back*2+len(trainPredict)+1:,:]=testPredict

testPredictPlot = pd.concat([nalst, trainPredict, nalst, testPredict],ignore_index=True)




plt.plot(dataset)
#plt.plot(trainPredictPlot)
#plt.plot(testPredict)
plt.plot(testPredictPlot)
plt.show()




