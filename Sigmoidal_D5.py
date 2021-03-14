import numpy as np
import matplotlib.pyplot as plt
import math 
import random

#PM10 CO Rainfall RH WD_HR (D5 by Sigmoidal model)

#data preprocesing:
#feature data
dataX=np.genfromtxt("dataset_X.csv",delimiter=',')
dataX=np.delete(dataX,[0,1,2,4,5,6,7,8,12,13,15,16,17],axis=1)
#target data
dataT=np.genfromtxt("dataset_T.csv",delimiter=',')
dataT=np.delete(dataT,[0],axis=1)

#shuffle the data to avoid the strange distribution
#concatenate the feature and target matrix and shuffle together
data_temp=np.c_[dataT,dataX]
np.random.shuffle(data_temp)
dataT=data_temp[:,0]
dataX=np.delete(data_temp,[0],axis=1)

def Sigmoidal(X):
    dataX_s=np.zeros(np.shape(X))
    for i in range(0,len(X)):
        for j in range(0,5):
            dataX_s[i,j]=math.exp(X[i,j])/(math.exp(X[i,j])+1)
    return dataX_s

def train_test_split(X,Y,test_size):
    X_train=np.array(X[:math.floor(len(X)*(1-test_size))])
    Y_train=np.array(Y[:math.floor(len(Y)*(1-test_size))])
    X_test=np.array(X[math.floor(len(X)*(1-test_size)):])
    Y_test=np.array(Y[math.floor(len(Y)*(1-test_size)):])
    Y_train=Y_train.reshape(len(Y_train),1)
    Y_test=Y_test.reshape(len(Y_test),1)
    return X_train, X_test, Y_train, Y_test

def linear_regression(X,Y):
    w=np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T,X)),X.T),Y)
    return w

def hypothesis(w,X):
    return np.matmul(w,np.transpose(X))

def rmse(a,b):
    return math.sqrt(np.sum((a-b)**2)/len(a))

dataX_s=Sigmoidal(dataX)
temp=np.ones((len(dataX_s),1))
dataX_s=np.c_[temp,dataX_s]

X_train,X_test,T_train,T_test = train_test_split(dataX_s,dataT,0.2)
w=linear_regression(X_train,T_train)

#plot the value of the model predict and the actual model (train part)
x=np.arange(0,len(X_train))
y=hypothesis(w.reshape(1,6),X_train).reshape(len(X_train),)
T_train=T_train.reshape(len(X_train),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="training_predict_value")
plt.plot(x,T_train,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_train,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Sigmoidal training D5")
plt.legend()
plt.show()

#plot the value of the model predict and the actual model (test part)
x=np.arange(0,len(X_test))
y=hypothesis(w.reshape(1,6),X_test).reshape(len(X_test),)
T_test=T_test.reshape(len(X_test),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="testing_predict_value")
plt.plot(x,T_test,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_test,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Sigmoidal testing D5")
plt.legend()
plt.show()