import numpy as np

def PCA(data, N):
    #去平均值
    meanVals = np.mean(data)
    meanRemoved = data - meanVals
    cov = np.cov(meanRemoved, rowvar=0) #协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(cov))  #求特征值
    index = np.argsort(eigVals)
    index = index[-1: -1 - N: -1]  #选择最大的N个特征值
    redEigVects = eigVects[ :,index]       #选取最大N个特征值对应的特征向量
    lowDData = meanRemoved.dot(redEigVects) #将数据转到新空间
    recon = (lowDData * redEigVects.T) + meanVals
    return lowDData, recon

#测试
x = np.array([
    [1, 3, 3],
    [2, 1, 4],
    [3, 5, 5]
])
d1, d2 = PCA(x, 2)
print(d1)
print(d2)