# -*- coding:utf8 -*-
import numpy as np
import math
from funcs import logisticFun, diffLogisticFun
from LayerOptions import getLayerOptions

options = getLayerOptions() # 获取全局配置信息
pDType = options['LayerParamDType'] # 网络层参数类型

class Layer:
    '''
    Layer Base Class of Neural Network 
    '''
    def __init__(self, iDim, hDim, oDim):
        '''
        iDim:输入维度大小
        hDim:隐层维度大小
        oDim:输出维度大小
        '''
        self.outData = None
        self.inputData = None
        self.iDim = iDim
        self.hDim = hDim
        self.oDim = oDim
        self.initWeights()
        
    def initWeights(self):
        '''
                    初始换权重
        '''
        sigma = math.sqrt(1.0/(self.iDim*1.0))
        self.weights = sigma*np.random.randn(self.hDim, self.iDim)
        self.weights = self.weights.astype(pDType)
        self.bias = np.ones((self.hDim), pDType)
    
    def setInput(self, inputData):
        self.inputData = inputData
    
    def runCalculate(self):
        '''
                    计算隐层状态
        '''
        if(self.weights.shape[1] != self.inputData.shape[0]):
            raise Exception('输入维度和输入矩阵维度不匹配 输入维度：'+str(self.inputData.shape[0])+" 矩阵维度："+str(self.weights.shape))
        multResu = np.dot(self.weights, self.inputData)
        self.mediaResu = multResu + self.bias
        self.outData = logisticFun(self.mediaResu) # 此处应该加一个激活函数
    
    def getOutput(self):
        return self.outData
    
    def backPro(self, error):
        '''
        error 应该是一个向量，其长度与隐层数量相同
        '''
        # 二次方程错误率计算
        # self.error = diffLogisticFun(self.mediaResu) * error
        
        # 交叉熵误差计算
        self.error = error
    
    def getError(self):
        '''
                    返回当前层的误差
        '''
        return np.dot(np.transpose(self.weights), self.error)
    
    def updateParams(self, lrate):
        '''梯度下降更新参数'''
        errorT = self.error.reshape((self.error.shape[0], 1))
        inputDataT = self.inputData.reshape((1, self.inputData.shape[0]))
        self.weights = self.weights - lrate*(np.dot(errorT, inputDataT))
        self.bias = self.bias - lrate*self.error

class InputLayer(Layer):
    def __init__(self, iDim, hDim, oDim):
        Layer.__init__(self, iDim, hDim, oDim)
    
    def runCalculate(self):
        Layer.runCalculate(self)
        