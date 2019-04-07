#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def normalEqation(x, y):
	'''
	使用正规方程法算法计算theta
	'''
	# 得到x的转置
	xTrains = x.transpose()
	m,n = np.shape(x)
	# theta 为 n维列向量
	theta = np.linalg.pinv(np.dot(xTrains,x)) 
	theta = np.dot(theta,xTrains)
	theta = np.dot(theta,y)
	return theta


# define the prepared 训练集
# the meaning of column : x1,x2,y
dataSet = np.array([
	[1.1,1.5,2.5],
	[1.3,1.9,3.2],
	[1.5,2.3,3.9],
	[1.7,2.7,4.6],
	[1.9,3.1,5.3],
	[2.1,3.5,6.0],
	[2.3,3.9,6.7],
	[2.5,4.3,7.4],
	[2.7,4.7,8.1],
	[2.9,5.1,8.8],
])


# print(dataSet)
m,n = np.shape(dataSet)
# print(m,n)
trainData = np.ones((m,n))
# 截取dataSet的前N-1列 
trainData[:,:-1] = dataSet[:,:-1]
# 获取dataSet的最后一列 
trainLabel = dataSet[:,-1]



theta = normalEqation(trainData, trainLabel)
print('thec value of theta is:')
print(np.round(theta,2))


# a test for the algorithm
x = np.array([
	[3.1, 5.5], 
	[3.3, 5.9], 
	[3.5, 6.3], 
	[3.7, 6.7], 
	[3.9, 7.1]
])


# define a predict function used to test
def predict(x,theta):
	m, n = np.shape(x)
	xTest = np.ones((m, n+1))
	xTest[:, :-1] = x
	yPre = np.dot(xTest,theta)
	return yPre

print('the predicted value is')
yP = predict(x, theta)
print(np.round(yP,2))

