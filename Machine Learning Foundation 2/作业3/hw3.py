# -*- coding:utf-8 -*-
import math
import numpy as np
import scipy.linalg as lin
# Problem 7
def p_u(u,v):
    return math.e**u+v*(math.e**(u*v))+2*u-2*v-3

def p_v(u,v):
    return 2*math.e**(2*v)+u*(math.e**(u*v))-2*u+4*v-2
def E(u,v):
    return math.e**u+math.e**(2*v)+math.e**(u*v)+u**2-2*u*v+2*(v**2)-3*u-2*v

def getE55(u,v):
    u_old,v_old=u,v
    u_new,v_new=0,0
    for i in range(1,6):
        u_new=u_old-0.01*p_u(u_old,v_old)
        v_new=v_old-0.01*p_v(u_old,v_old)
        print(u_new,v_new)
        u_old,v_old=u_new, v_new
    
    return E(u_new,v_new)


# print(getE55(0, 0))  # 2.8250003566832635

# Problem 10

def getHessianMatrix(u,v):
    a11=math.e**u+(v**2)*(math.e**(u*v))+2
    a12=math.e**(u*v)+u*v*(math.e**(u*v))-2
    a21=a12
    a22=4*math.e**(2*v)+(u**2)*(math.e**(u*v))+4

    return np.array([[a11,a12],[a21,a22]])

def oneVector(u,v):
    a1=p_u(u,v)
    a2=p_v(v,v)
    return np.mat([[a1],[a2]])

def getE55_2(u, v):
    uv=np.array([[float(u)],[float(v)]])
    for i in range(5):
        du = math.exp(u)+v*math.exp(u*v)+2*u-2*v-3
        dv = 2*math.exp(2*v)+u*math.exp(u*v)-2*u+4*v-2

        duu = math.exp(u)+(v**2)*math.exp(u*v)+2
        dudv = math.exp(u*v)+(u*v)*math.exp(u*v)-2
        dvv = 4*math.exp(2*v)+(u**2)*math.exp(u*v)+4

        ddE=np.array([[duu,dudv],[dudv,dvv]])
        dE=np.array([du,dv])

        uv=uv-lin.inv(ddE)*dE
        u=uv[0,0]
        v=uv[1,0]

    print(u,v)
    return E(u,v)

        
# print(getE55_2(0, 0))  # u=0.6138669497141045 v=0.07924631001569933  E(5,5)=2.361160409666252


# Problem 13-15

def generateData(num):
    X1=np.random.uniform(-1,1,num)
    X2=np.random.uniform(-1,1,num)
    tempX=np.c_[X1,X2]
    X=np.c_[np.ones((num,1)),tempX]
    tempY=np.sign(X1**2+X2**2-0.6)
    tempY[tempY==0]=-1

    positions=np.random.permutation(num)
    tempY[positions[0:round(0.1*num)]]*=-1
    Y=tempY.reshape(num,1)
    return X,Y

def p13_main():
    X,Y=generateData(1000)
    total=0
    for i in range(1000):
        theta=lin.pinv(X.T.dot(X)).dot(X.T).dot(Y)
        y_pred=np.sign(X.dot(theta))
        y_pred[y_pred==0]=-1
        err = np.sum(y_pred != Y)/len(Y)
        total+=err
    return total/1000

# res=p13_main() 
# print(res)  # 0.5350000000000092上下

def transformData(num):
    X,Y=generateData(num)
    rows, cols = X.shape
    back = np.zeros((rows, 6))
    back[:, 0:cols] = X
    back[:, cols] = X[:, 1]*X[:, 2]
    back[:, cols+1] = X[:, 1]**2
    back[:, cols+2] = X[:, 2]**2
    trasX = back
    return trasX,Y

def p14_main():
    total = 0
    for i in range(1000):
        trasX, Y = transformData(1000)
        theta = lin.pinv(trasX.T.dot(trasX)).dot(trasX.T).dot(Y)
        y_pred = np.sign(trasX.dot(theta))
        y_pred[y_pred == 0] = -1
        err = np.sum(y_pred != Y)/len(Y)
        total += err
        print(theta.T)
    return total/len(Y)

# res=p14_main()
# print(res)

# # theta=[[-0.95617173  0.0380101  -0.03091189 -0.01252074  1.46113224  1.49864974]]
# # err_rate = 0.12375999999999998

# Problem 18-20

def load_data(file):
    m = []
    with open(file) as f:
        for line in f:
            m.append([float(i) for i in line.split()])
    mm = np.asarray(m)
    rows = len(mm)
    X = mm[:, :-1]
    X=np.c_[np.ones((rows,1)),X]
    Y = mm[:, -1]
    Y=Y.reshape((rows,1))
    return X, Y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression(X,Y,eta,iter_nums,flag):
    '''
    data:(X,Y)
    eta:学习率
    item_nums:迭代次数
    flag：如果flag=False,采用gradient descent
          如果falg=True,则采用SGD
    '''
    rows,cols=X.shape
    print(X.shape)
    print(Y.shape)
    theta=np.zeros((cols,1))
    num=0

    for i in range(iter_nums):
        if not flag:
            #derr=(-1*X*Y).T.dot(sigmoid(-1*X.dot(theta)*Y))/rows
            derr = (-1*X*Y).T.dot(sigmoid(-1*X.dot(theta)*Y))/rows
        else:
            if num>=rows:
                num=0
            derr = -Y[num, 0]*X[num: num+1, :].T * sigmoid(-1*X[num, :].dot(theta)[0]*Y[num, 0])
            num+=1
        theta-=eta*derr

    return theta

def evaluate(X,Y,theta):
    Y_pred=X.dot(theta)
    Y_pred[Y_pred>0]=1
    Y_pred[Y_pred<=0] = -1
    err=np.sum(Y_pred!=Y)/len(Y)
    return err


# P18
X, Y = load_data(
    '/Users/yanmk/学习/技术学习/人工智能/机器学习/Machine Learning Foundations/Machine Learning Foundation 2/作业3/hw3_train.dat')
# eta=0.001

eta=0.01

T=2000
flag=False
theta=logistic_regression(X,Y,eta,T,flag)
Ein=evaluate(X,Y,theta)
print('Ein:',Ein)
Xtest, Ytest = load_data(
    '/Users/yanmk/学习/技术学习/人工智能/机器学习/Machine Learning Foundations/Machine Learning Foundation 2/作业3/hw3_test.dat')
Eout=evaluate(Xtest,Ytest,theta)
print('Eout:',Eout)

# P18
# Ein: 0.466
# Eout: 0.475

# P19

# Ein: 0.197
# Eout: 0.22


#20:

# flag=True
# eta=0.001
# theta = logistic_regression(X, Y, eta, T, flag)
# Ein = evaluate(X, Y, theta)
# print('Ein:', Ein)
# Xtest, Ytest = load_data(
#     '/Users/yanmk/学习/技术学习/人工智能/机器学习/Machine Learning Foundations/Machine Learning Foundation 2/作业3/hw3_test.dat')
# Eout = evaluate(Xtest, Ytest, theta)
# print('Eout:', Eout)

# # Ein: 0.464
# # Eout: 0.473





    
