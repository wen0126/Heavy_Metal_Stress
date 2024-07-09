#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'jinhuaan@imde.ac.cn'
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import func as fc
from sklearn.externals import joblib


# load data
data = pd.read_csv(r"ANN.csv")

X = data.iloc[:, 1:].values.astype('float32')

Y = data.iloc[:, 0].values.astype('float32')
Y = Y.reshape(len(Y),1)


MinValue = min(Y) 
MaxValue = max(Y) 

print(data.shape)
print(X.shape)
print(Y.shape)

# data normalization
mms = preprocessing.MinMaxScaler()
X_mms = mms.fit_transform(X)
Y_mms = mms.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_mms, Y_mms, test_size=0.2, random_state=100)  
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
################

nn = MLPRegressor(activation='relu',
                  hidden_layer_sizes=(11,),
                  learning_rate='adaptive',
                  max_iter=200)

nn.fit(X_train,Y_train.ravel())
joblib.dump(nn,'ANN.model')
Training = joblib.load('ANN.model')
Y_sim = Training.predict(X_train)
Y_sim = mms.inverse_transform(Y_sim.reshape(len(Y_sim),1))
Y_pre = Training.predict(X_test)
Y_pre = mms.inverse_transform(Y_pre.reshape(len(Y_pre),1))
Y_test = mms.inverse_transform(Y_test.reshape(len(Y_test),1))
Y_train = mms.inverse_transform(Y_train.reshape(len(Y_train),1))



#################
R1 = fc.compute_correlation(Y_sim, Y_train)  
R2 = fc.compute_correlation(Y_pre, Y_test)  
RMSE1 = fc.compute_rmse(Y_sim, Y_train)  
RMSE2 = fc.compute_rmse(Y_pre, Y_test)  
print(R1 ** 2, RMSE1, R2 ** 2, RMSE2)

plt.figure(num = 2)
loc_x =  0.3  
loc_y1 = 0.6  
loc_y2 = 1.0  
fsize =  12   

plt.subplot(1, 2, 1)    
plt.scatter(Y_train, Y_sim,s=10,c='',marker='.',edgecolors='g')
plt.xlabel('Train values')
plt.ylabel('Simulation values')

plt.plot([MinValue, MaxValue], [MinValue, MaxValue], color='k', linewidth=1.0)

plt.xlim((MinValue, MaxValue))
plt.ylim((MinValue, MaxValue))

plt.text(MinValue+loc_x, MaxValue-loc_y1-5, 'R2 = ''%.2f' % R1**2, size = fsize)    
plt.text(MinValue+loc_x, MaxValue-loc_y2-10, 'RMSE = ''%.2f' % RMSE1, size = fsize)  

plt.show()

plt.subplot(1, 2, 2)    
plt.scatter(Y_test, Y_pre,s=10,c='',marker='.',edgecolors='m')
plt.xlabel('Test values')
plt.ylabel('Prediction values')

plt.plot([MinValue, MaxValue], [MinValue, MaxValue], color='k', linewidth=1.0)

plt.xlim((MinValue, MaxValue))
plt.ylim((MinValue, MaxValue))

plt.text(MinValue+loc_x, MaxValue-loc_y1-5, 'R2 = ''%.2f' % R2**2, size = fsize)   
plt.text(MinValue+loc_x, MaxValue-loc_y2-10, 'RMSE = ''%.2f' % RMSE2, size = fsize) 

plt.show()