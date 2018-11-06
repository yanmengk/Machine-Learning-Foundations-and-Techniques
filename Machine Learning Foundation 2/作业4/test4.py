# -*- coding:utf-8 -*-

import math
import numpy as np
from scipy.optimize import fsolve
import scipy.linalg as lin

# Problem 9
print(2*math.exp(-2*0.0001*10000))  # 0.2706705664732254


# Problem 13-20
def load_data(file):
    m = []
    with open(file) as f:
        for line in f:
            m.append([float(i) for i in line.split()])
    mm = np.asarray(m)
    rows = len(mm)
    X = mm[:, :-1]
    X = np.c_[np.ones((rows, 1)), X]
    Y = mm[:, -1]
    Y = Y.reshape((rows, 1))
    return X, Y

def linear_regression_reg(X,Y,lamda):
    return lin.pinv(X.T.dot(X)+lamda*np.eye(3)).dot(X.T).dot(Y)

def error_evaluate(y_preds,y_trues):
    y_preds[y_preds>0]=1
    y_preds[y_preds<=0]=-1
    result=np.sum(y_preds!=y_trues)/len(y_trues)
    return result

X, Y = load_data(
    '/Users/yanmk/学习/技术学习/人工智能/机器学习/Machine Learning Foundations/Machine Learning Foundation 2/作业4//hw4_train.dat')
X_test, Y_test = load_data(
    '/Users/yanmk/学习/技术学习/人工智能/机器学习/Machine Learning Foundations/Machine Learning Foundation 2/作业4//hw4_test.dat')

# Problem 13

# lamda=10
# w=linear_regression_reg(X,Y,lamda)
# #计算训练集误差 Ein
# y_train=X.dot(w)
# Ein=error_evaluate(y_train,Y)
# print('Ein:',Ein)
# #计算测试集误差 Eout

# y_test=X_test.dot(w)
# Eout=error_evaluate(y_test,Y_test)
# print('Eout:',Eout)
# # Ein: 0.05
# # Eout: 0.045

# Problem 14-15

# for i in range(2, -11, -1):
#     lamda=10**i
#     w=linear_regression_reg(X, Y, lamda)
#     y_train=X.dot(w)
#     Ein=error_evaluate(y_train,Y)
#     print('Ein:',i,Ein)

#     y_test=X_test.dot(w)
#     Eout=error_evaluate(y_test,Y_test)
#     print('Eout:',i,Eout)
#     print('=============')

# Problem 16-17

# X_train=X[:120]
# Y_train=Y[:120]
# X_val=X[120:200]
# Y_val=Y[120:200]

# for i in range(2, -11, -1):
#     lamda=10**i
#     w=linear_regression_reg(X_train, Y_train, lamda)
#     y_train=X_train.dot(w)
#     Ein=error_evaluate(y_train,Y_train)
#     print('Ein:',i,Ein)

#     y_val=X_val.dot(w)
#     Eval=error_evaluate(y_val,Y_val)
#     print('Eval',i,Eval)

#     y_test=X_test.dot(w)
#     Eout=error_evaluate(y_test,Y_test)
#     print('Eout:',i,Eout)
#     print('=============')

# problem 18

# lamda=1
# w=linear_regression_reg(X,Y,lamda)
# #计算训练集误差 Ein
# y_train=X.dot(w)
# Ein=error_evaluate(y_train,Y)
# print('Ein:',Ein)
# #计算测试集误差 Eout

# y_test=X_test.dot(w)
# Eout=error_evaluate(y_test,Y_test)
# print('Eout:',Eout)
# # # Ein: 0.035
# # # Eout: 0.02


# Problem 19

#5折交叉验证
for j in range(2, -11, -1):
    lamda=10**j
    totalErr=0
    for i in range(5):
        X_val=X[40*i:40*(i+1)]
        Y_val =Y[40*i:40*(i+1)]
        X_train=np.r_[X[0:40*i],X[40*(i+1):]]
        Y_train = np.r_[Y[0:40*i], Y[40*(i+1):]]
        w=linear_regression_reg(X_train,Y_train,lamda)
        y_val=X_val.dot(w)
        totalErr+=error_evaluate(y_val,Y_val)
    print('Eval',j,totalErr/5)

# # Problem 20
# lamda=10**(-8)
# w=linear_regression_reg(X,Y,lamda)
# #计算训练集误差 Ein
# y_train=X.dot(w)
# Ein=error_evaluate(y_train,Y)
# print('Ein:',Ein)
# #计算测试集误差 Eout

# y_test=X_test.dot(w)
# Eout=error_evaluate(y_test,Y_test)
# print('Eout:',Eout)
# # # Ein: 0.015
# # # Eout: 0.02
