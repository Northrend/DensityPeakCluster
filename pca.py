#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# zero mean
def zeroMean(dataMat):      
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData,meanVal


def pca(dataMat,percentage=0.99):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)    # get covariance mat, return ndarray
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))  # get eigen value and vector
    n=percentage2n(eigVals,percentage)              # get the former n vectors
    eigValIndice=np.argsort(eigVals)            # 对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
    return lowDDataMat,reconMat


if __name__ == "__main__":
    in_np
