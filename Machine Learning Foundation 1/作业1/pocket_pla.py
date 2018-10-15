# -*- coding:utf-8 -*-
import numpy as np 
import random

# 载入数据 hw1_15_train.dat
def load_data(file):
    m=[]
    with open(file) as f:
        for line in f:
            m.append([float(i) for i in line.split()])
    mm=np.asarray(m)
    rows=len(mm)
    X=np.c_[np.ones(rows),mm[:,:-1]]
    Y=mm[:,-1]
    return X,Y

# sign函数
def sign(x):
    if x>0:
        return 1
    else:
        return -1

# 衡量在w某一固定值下的错误率error rate
def evaluate(X,Y,w):
    n=len(Y)
    errors=sum([1 for i in range(n) if sign(np.dot(X[i],w))!=Y[i]])
    return errors/n

# Pocket Algorithm on PLA核心代码
def pla_pocket(X,Y,updates=50,pocket=True):
    '''
    w:返回迭代过后的w值
    '''
    n=len(X)
    cols = len(X[0])
    w=np.zeros(cols)
    wg=w #wg 表示存储的当前最佳的w的值
    error=evaluate(X,Y,w)

    for k in range(updates):
        idx=random.sample(range(n),n)
        for i in idx:
            if sign(np.dot(X[i], w)) != Y[i]:
                w=w+Y[i]*X[i]
                e=evaluate(X,Y,w)
                if e<error:
                    error=e 
                    wg=w
                break
    if pocket:
        return wg
    return w

def main(updates,pocket):
    X, Y = load_data('hw1_18_train.dat')
    X_test, Y_test = load_data('hw1_18_test.dat')
    errors=0
    n=2000 # 迭代次数
    for i in range(n):
        w=pla_pocket(X,Y,updates=updates,pocket=pocket)
        errors+=evaluate(X_test,Y_test,w)
    return errors/n


if __name__=="__main__":
    # Question 18
    print(main(updates=50, pocket=True))  # 0.13296599999999956

    # Question 19
    print(main(updates=50, pocket=False))  # 0.35915600000000036

    # Question 20
    print(main(updates=100, pocket=True))  # 0.1159760000000001







