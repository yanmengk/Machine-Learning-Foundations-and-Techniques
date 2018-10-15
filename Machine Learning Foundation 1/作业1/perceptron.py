# -*- coding:utf-8 -*-
import numpy as np 
import random

# 载入数据 hw1_15_train.dat
def load_data():
    m=[]
    with open('hw1_15_train.dat') as f:
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

# Perceptron Learning Algorithm核心代码
def pla(X,Y,rand=False,alpha=1):
    '''
    ans:记录返回更新的次数
    '''
    n=len(X)
    cols = len(X[0])
    w=np.zeros(cols)
    ans=0 # ans记录更新的次数

    idx=range(n)
    if rand:
        idx=random.sample(idx,n)
    
    k=0
    update=False
    while True:
        i=idx[k]
        if sign(np.dot(X[i],w))!=Y[i]:
            ans+=1
            w=w+alpha*Y[i]*X[i]
            update=True
        k+=1
        if k==n:
            if update==False:
                break
            k=0
            update=False
    return ans

def naive_cycle():
    X,Y=load_data()
    ans=pla(X,Y)
    return ans 

def random_cycle(n,alpha=1):
    X, Y = load_data()
    cnt=0
    for i in range(n):
        cnt+=pla(X,Y,rand=True,alpha=alpha)
    return cnt/n


if __name__=="__main__":
    #Question 15答案
    print(naive_cycle()) # 45
    print(random_cycle(2000))  # 40.344
    print(random_cycle(2000, 0.5))  # 39.8525





