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

#D5
k=6
for i in range(1,6):
    for j in range(1,i+1):
        if k in range(6,21):
           dataX=np.insert(dataX,k,values=dataX[:,i]*dataX[:,j],axis=1)
            k+=1

#D12
k=13
for i in range(1,13):
    for j in range(1,i+1):
        if k in range(13,91):
           dataX=np.insert(dataX,k,values=dataX[:,i]*dataX[:,j],axis=1)
            k+=1 


#append the dataX to match the theta (171 features)
k=18
for i in range(1,18):
    for j in range(1,i+1):
        if k in range(18,171):
            dataX=np.insert(dataX,k,values=dataX[:,i]*dataX[:,j],axis=1)
            k+=1

#target data
dataT=np.genfromtxt("dataset_T.csv",delimiter=',')
dataT=np.delete(dataT,[0],axis=1)


#shuffle the data to avoid the strange distribution
#concatenate the feature and target matrix and shuffle together
data_temp=np.c_[dataT,dataX]
np.random.shuffle(data_temp)
dataT=data_temp[:,0]
dataX=np.delete(data_temp,[0],axis=1)

#split the data into training set and testing set
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

def rmse(a,b):
    return math.sqrt(np.sum((a-b)**2)/len(a))

def hypothesis(w,X):
    return np.matmul(w,np.transpose(X))

X_train,X_test,T_train,T_test = train_test_split(dataX,dataT,0.2)
w=linear_regression(X_train,T_train)

