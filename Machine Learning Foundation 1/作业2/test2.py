# -*- coding:utf-8 -*-

# Problem 3
import math
from scipy.optimize import fsolve

def func(N):
    return 4*((2*N)**10)*math.e**(-1/8*0.05*0.05*N)


# print(func(420000))  # 697.75
# print(func(440000))  # 2.14

# print(func(460000))  # 0.0065

# print(func(480000))  # 1.91e-05

# Problem 4
def p4_1(d,f):
    return math.sqrt(8/10000*math.log(4*((2*10000)**d)/f))
def p4_2(d,f):
    return math.sqrt(2*math.log(2*10000*((10000)**d))/10000)+math.sqrt(2/10000*math.log(1/f))+1/10000




def p4_5(d,f):
    return math.sqrt(16/10000*(math.log(2*((10000)**d)/math.sqrt(f))))


def p4_3(x):
    d=50
    f=0.05
    res=float(x[0])
    return [res-math.sqrt(1/10000*(2*res+math.log(6*(2*10000)**d/f)))]


# print(p4_1(50, 0.05)) # 0.6322
# print(p4_2(50, 0.05))  # 0.3313
# print(p4_5(50, 0.05))  # 0.8604
# print(fsolve(p4_3, [1]))  # [0.22369829]

## ????????
def p4_4(x):
    d=50
    f=0.05
    res=float(x[0])
    return [res-math.sqrt(1/2/10000*(4*res*(1+res)+math.log(4*(10000)**(2*d)/f)))]


print(fsolve(p4_4, [1]))

