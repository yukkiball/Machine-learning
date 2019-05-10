import numpy as np


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0], float(lineArr[10]))])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights += alpha * dataMatrix.transpose() * error
    return weights

def stocTradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for i in range(numIter):
        dataIndex = range(m)
        for j in range(m):
            alpha = 4 / (1 + i + j) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(dataMatrix[randIndex] * weights)
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(inX * weights)
    if prob > 0.5:
        return 1
    else:
        return 0


def classify(filename, weights):

    data, label = loadDataSet(filename)
    dataMat = np.mat(data)
    labelMat = np.mat(label)
    m = np.shape(dataMat)[0]
    error = 0
    prob = sigmoid(dataMat * weights)
    for i in range(m):

        if prob[i] > 0.5:
            print(1)
            if labelMat[i] != 1:
                error += 1
        else:
            print(0)
            if labelMat[i] != 0:
                error += 1
    print("error:", error / m)