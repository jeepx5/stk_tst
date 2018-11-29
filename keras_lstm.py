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

#for ADF check
from statsmodels.tsa.stattools import adfuller
#for KPSS check
from statsmodels.tsa.stattools import kpss


df = pd.read_csv('./dataset/international-airline-passengers.csv',
                      usecols = [1], engine='python', skipfooter = 3)

dataset = df.values
dataset = dataset.astype('float32')
print(df.head(5))
df['Month'] = pd.date_range('1/1/1949', periods = len(dataset), freq = 'M')
print(df.head(5))

#data preprocessing
dtrain = df
dtrain.timestamp = pd.to_datetime(dtrain.Month, format = '%Y-%m')
dtrain.index= dtrain.timestamp
dtrain.drop('Month', axis = 1, inplace = True)
#dtrain.rename(columns={'$112': 'Passenger'}, inplace = True)
dtrain.columns = ['passenger']
print(dtrain.head(5))

'''
结果1：两种检测均得出结论：序列是非平稳的->序列是非平稳的

结果2：两种检测均得出结论：序列是平稳的->序列是平稳的

结果3：KPSS =平稳；ADF =非平稳->趋势平稳，去除趋势后序列严格平稳

结果4：KPSS =非平稳；ADF =平稳->差分平稳，利用差分使序列平稳。
'''



# define function for ADF test
def adf_test(timeseries):
    #perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag= 'AIC')
    dfoutput = pd.Series(dftest[0:4])#,index = ['Test Statistic of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key]=value
    print(dfoutput)

adf_test(dtrain['passenger'])

#ADF检验结果：ADF检验的统计量为1%，p值为5%，临界值为10%，置信区间为10%。
'''f="">平稳性检验：如果检验统计量小于临界值，
我们可以否决原假设(也就是序列是平稳的)。
当检验统计量大于临界值时，否决原假设失败(这意味着序列不是平稳的)。'''

# define function for kpss test
def kpss_test(timesereies):
    print('results of kpss test')
    kpsstest = kpss(timesereies, regression = 'c')
    kpss_output = pd.Series(kpsstest[0:3])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] =  value
    print(kpss_output)

kpss_test(dtrain['passenger'])

'''
KPSS检验结果：KPSS检验-检验统计量、
p-值和临界值和置信区间分别为1%、2.5%、5%和10%'''
'''f="">平稳性检验：如果检验统计量大于临界值，
则否决原假设(序列不是平稳的)。
如果检验统计量小于临界值，
则不能否决原假设(序列是平稳的)。'''


#符合结果4

#时间序列的平稳化
'''变换用于对方差为非常数的序列进行平稳化。
常用的变换方法包括幂变换、平方根变换和对数变换。
对飞机乘客数据集进行快速对数转换和差分：'''
look_back =7
dtrain['passenger_diff'] = dtrain['passenger']-dtrain['passenger'].shift(look_back)
dtrain['passenger_diff'].dropna().plot()

plt.show()

dtrain['passenger_log'] = np.log(dtrain['passenger'])
dtrain['passenger_log_diff'] = dtrain['passenger_log']-dtrain['passenger_log'].shift(look_back)
dtrain['passenger_log_diff'] = np.sqrt(dtrain['passenger_log_diff'].dropna())
dtrain['passenger_log_diff'].plot()
plt.show()

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


#look_back =3

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
