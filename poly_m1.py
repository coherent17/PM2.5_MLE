import numpy as np
import matplotlib.pyplot as plt
import math 
import random

#data preprocesing:
#feature data
dataX=np.genfromtxt("dataset_X.csv",delimiter=',')
dataX=np.delete(dataX,[0],axis=1)
temp=np.array([1]*len(dataX))
dataX=np.c_[temp,dataX]
#target data
dataT=np.genfromtxt("dataset_T.csv",delimiter=',')
dataT=np.delete(dataT,[0],axis=1)

#shuffle the data to avoid the strange distribution
np.random.shuffle(dataX)
np.random.shuffle(dataT)

#split the data into training set and testing set
def train_test_split(X,Y,test_size):
    X_train=np.array(X[:math.floor(len(X)*(1-test_size))])
    Y_train=np.array(Y[:math.floor(len(Y)*(1-test_size))])
    X_test=np.array(X[math.floor(len(X)*(1-test_size)):])
    Y_test=np.array(Y[math.floor(len(Y)*(1-test_size)):])
    Y_train=Y_train.reshape(len(Y_train),1)
    Y_test=Y_test.reshape(len(Y_test),1)
    return X_train, X_test, Y_train, Y_test

X_train,X_test,T_train,T_test = train_test_split(dataX,dataT,0.2)

def linear_regression(X,Y):
    w=np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)
    return w

def rmse(a,b):
    return math.sqrt(np.sum((a-b)**2)/len(a))

def hypothesis(w,X):
    return np.matmul(w,np.transpose(X))

w=linear_regression(X_train,T_train)

#plot the value of the model predict and the actual model (train part)
x=np.arange(0,len(X_train))
y=hypothesis(w.reshape(1,18),X_train).reshape(len(X_train),)
T_train=T_train.reshape(len(X_train),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="training_predict_value")
plt.plot(x,T_train,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_train,y)))
plt.show()




#plot the weight of each features
# x=np.arange(1,19).reshape(18,1)

# plt.plot(x,w,'b.')
# plt.show()

