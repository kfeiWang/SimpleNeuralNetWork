# -*- coding:utf8 -*-
from inputLayer import Layer

import numpy as np
from funcs import crossEntryCostFun, squareCostFun

iLayer = Layer(10, 20, 15) # input layer
hLayer = Layer(20, 20, 15) # hidden layer
oLayer = Layer(20, 10, 10) # output layer

inputData = np.ones((10), 'int32') # 输入数据
tarData = np.ones((10), 'int32') # 目标
epoch = 100000
lrate = 0.2
showEpoch = 500

for i in range(epoch):
    iLayer.setInput(inputData)
    iLayer.runCalculate()
    midResu1 = iLayer.getOutput()
    
    hLayer.setInput(midResu1)
    hLayer.runCalculate()
    midResu2 = hLayer.getOutput()
    
    oLayer.setInput(midResu2)
    oLayer.runCalculate()
    outputData = oLayer.getOutput()
    
    # 一个样本更新一次参数，可以多个样本更新一次参数
    # 计算输出层误差
    # cost = 0.5*np.square(outputData - tarData)
    cost = crossEntryCostFun(outputData, tarData)
    if i%showEpoch == 0:
        print 'costSquare',np.sum(cost)
    error = (outputData - tarData)
    oLayer.backPro(error)
    # 更新输出层参数
    oLayer.updateParams(lrate)
    # 计算隐层误差
    hLayer.backPro(oLayer.getError())
    # 更新隐层参数
    hLayer.updateParams(lrate)
    # 计算输入层误差
    iLayer.backPro(hLayer.getError())
    # 更新输入层参数
    iLayer.updateParams(lrate)
    # 继续下一轮
    if i%1000 == 0:
        if lrate > 0.01:
            lrate = lrate/2.0

print 'inputData',inputData
#print midResu1
# print midResu2
print 'outputData',outputData
print 'tarData',tarData
errors = 0.5*np.square(outputData - tarData)
print 'errors', errors
print 'cost', np.sum(errors)