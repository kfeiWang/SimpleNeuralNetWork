# -*- coding:utf8 -*-
import numpy as np

def logisticFun(x):
    '''
    logistic函数
    '''
    return 1.0/(1 + np.exp(-1.0*x))

def diffLogisticFun(x):
    '''
    logistic函数的导函数
    '''
    return logisticFun(x)*(1 - logisticFun(x))

def crossEntryCostFun(x, y):
    '''交叉熵代价函数'''
    return -(y*np.log2(x)+(1-y)*np.log2(1-x))

def squareCostFun(x, y):
    '''二次代价函数'''
    return 0.5*np.square(x - y)