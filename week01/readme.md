# 第一周机器学习

## 梯度下降法的Python实现
参考代码，自己就一些细节进行优化
https://www.cnblogs.com/focusonepoint/p/6394339.html
```Python
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from numpy import linalg



def gradientDescent(x, y, theta, m, alpha, maxIteration):
	'''
	使用批处理梯度下降算法计算theta
	'''
	# 得到x的转置
	# 即 x的第一行为x1 第二行为x2 第三行全部初始化为1
	xTrains = x.transpose()
	# theta 是一个列向量
	for i in xrange(0,maxIteration):
		# x矩阵(10*3)与theta(3*1)矩阵相乘
		# hypothesis(i) = x1(i)*theta1(i) + x2(i)*theta2(2) + 1*theta0(i)
		hypothesis = np.dot(x, theta)
		# 作差
		loss = hypothesis - y
		# 当loss的范数在我们的误差允许范围内 就停止循环
		if (linalg.norm(loss) < 1e-5):
			break
		# xTrains (3*10) * loss(10*1) = gradient(3*1)
		# 计算代价函数
		gradient = (1.0/m) * np.dot(xTrains, loss)
		theta = theta - alpha * gradient
	print('the number of iteration is %d' % i);
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

# print(m,n)
theta = np.ones(n)
# print(theta)
alpha = 0.001

# the max time of iteration 这个值定义的尽量大(考虑计算机的性能)
maxIteration = 10000000
theta = gradientDescent(trainData, trainLabel, theta, m, alpha, maxIteration)
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


```
运行结果
```
the number of iteration is 114575
thec value of theta is:
[ 0.71  1.39 -0.38]
the predicted value is
[ 9.5 10.2 10.9 11.6 12.3]
[Finished in 2.2s]
```