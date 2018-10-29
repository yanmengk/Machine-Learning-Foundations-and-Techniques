# -*- coding:utf-8 -*-

# Problem 3
import math
import numpy as np
from scipy.optimize import fsolve

def func(N):
    return 4*((2*N)**10)*math.e**(-1/8*0.05*0.05*N)


# print(func(420000))  # 697.75
# print(func(440000))  # 2.14

# print(func(460000))  # 0.0065

# print(func(480000))  # 1.91e-05

# Problem 4
def p4_1(d,f,N):
    return math.sqrt(8/N*math.log(4*((2*N)**d)/f))
def p4_2(d,f,N):
    return math.sqrt(2*math.log(2*N*((N)**d))/N)+math.sqrt(2/N*math.log(1/f))+1/N


def p4_3(x):
    d = 50
    f = 0.05
    #N=10000
    N=5


    res = float(x[0])
    return [res-math.sqrt(1/N*(2*res+math.log(6*(2*N)**d/f)))]


def p4_4(x):
    d = 50
    f = 0.05
    #N=10000
    N=5


    res = float(x[0])
    #a = math.log(4*(10000)**100/f)
    return [res-math.sqrt(1/(2*N)*(4*res*(1+res) + math.log(4/f)+100*math.log(N)))]

def p4_5(d,f,N):
    return math.sqrt(16/N*(math.log(2*((N)**d)/math.sqrt(f))))




#Q4

# print(p4_1(50, 0.05,10000)) # 0.6322
# print(p4_2(50, 0.05,10000))  # 0.3313
# print(fsolve(p4_3, [1]))  # 0.22369829

# print(fsolve(p4_4, [1]))  # 0.21522805

# print(p4_5(50, 0.05,10000))  # 0.8604

#Q5

print(p4_1(50, 0.05,5)) # 0.6322
print(p4_2(50, 0.05,5))  # 0.3313
print(fsolve(p4_3, [1]))  # 0.22369829

print(fsolve(p4_4, [1]))  # 0.21522805

print(p4_5(50, 0.05,5))  # 0.8604

# Problem 17-18
def generateData():
    x=np.random.uniform(-1,1,20)
    x=np.sort(x) #x按从小到大排列
    y=np.sign(x)
    y[y==0]=-1
    prob=np.random.uniform(0,1,20)
    y[prob>=0.8] *= -1
    return x,y

#单维决策树桩算法
def decision_stump(x,y):

    thetas = np.array([float("-inf")]+[(x[i]+x[i+1])/2 for i in range(0, x.shape[0]-1)]+[float("inf")])
    nums=len(x)
    Ein=1
    s=1
    target_theta=0.0


    for theta in thetas:
        y_pos=np.where(x>theta,1,-1)
        y_neg=np.where(x<theta,1,-1)
        error_pos_nums=sum(y_pos!=y)
        error_neg_nums=sum(y_neg!=y)
        if error_pos_nums>error_neg_nums:
            current=error_neg_nums/nums
            if current<Ein:
                s=-1
                Ein=current
                target_theta=theta
        else:
            current2=error_pos_nums/nums
            if current2<Ein:
                s=1
                Ein=current2
                target_theta=theta
    
    if target_theta==float('-inf'):
        target_theta=-1.0
    if target_theta==float('inf'):
        target_theta=1.0
    return Ein,s,target_theta


def decision_stump2(x, y):
    #需要先对x进行排序
    thetas = np.array([float("-inf")]+[(x[i]+x[i+1]) /
                                       2 for i in range(0, x.shape[0]-1)]+[float("inf")])
    nums = len(x)
    Ein = 1
    s = 1
    target_theta = 0.0

    for theta in thetas:
        y_pos = np.where(x > theta, 1, -1)
        y_neg = np.where(x < theta, 1, -1)
        error_pos_nums = sum(y_pos != y)
        error_neg_nums = sum(y_neg != y)
        if error_pos_nums > error_neg_nums:
            current = error_neg_nums/nums
            if current < Ein:
                s = -1
                Ein = current
                target_theta = theta
        else:
            current2 = error_pos_nums/nums
            if current2 < Ein:
                s = 1
                Ein = current2
                target_theta = theta

    return Ein, s, target_theta

def load_data(file):
    m = []
    with open(file) as f:
        for line in f:
            m.append([float(i) for i in line.split()])
    mm = np.asarray(m)
    rows = len(mm)
    X = mm[:, :-1]
    Y = mm[:, -1]
    return X, Y


def calculate_ds_multi(x,y,x_test,y_test):
    rows,cols=x.shape # rows=100,cols=9
    Ein=1
    theta=0
    s=1
    index=0
    for i in range(cols):
        input_x=x[:,i]
        input_data = np.transpose(np.array([input_x, y])) #np.array([input_x, y])形状为(2,100)
        input_data=input_data[np.argsort(input_data[:,0])]
        curr_Ein,curr_s,curr_theta=decision_stump2(input_data[:,0],input_data[:,1])
        if curr_Ein<Ein:
            Ein=curr_Ein
            s=curr_s
            theta=curr_theta
            index=i
    
    y_pred = s*np.sign(X_test[:,index]-theta)
    y_pred[y_pred == 0] = -1
    test_row, test_col = x_test.shape
    Eout = np.sum(y_pred != y_test.reshape(test_row))/len(y_pred)
    return Ein,Eout




        
# if __name__=="__main__":
#     # Q17-18

#     T=5000
#     total_Ein = 0
#     total_Eout = 0
#     for t in range(T):
#         x,y=generateData()
#         curr_Ein,s,target_theta=decision_stump(x,y)
#         total_Ein+=curr_Ein
#         total_Eout+=(0.5+0.3*s*(abs(target_theta)-1))
#     print(total_Ein/T)
#     print(total_Eout/T)

#     #Q19-20
#     X, Y = load_data('hw2_train.dat')
#     X_test, Y_test = load_data('hw2_test.dat')

#     Ein,Eout=calculate_ds_multi(X,Y,X_test,Y_test)
#     print("训练集误差Ein：", Ein)
#     print('测试集误差Eout: ', Eout)



        





    




