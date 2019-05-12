import numpy as np

#决策树桩决策
def stumpClassify(data, dimen, threshVal, threshIneq):
    retArray = np.ones(data.shape[0])
    if threshIneq == 'lt':
        retArray[data[:, dimen] <= threshVal] = -1.0
    else:
        retArray[data[:, dimen] > threshVal] = -1.0
    return retArray

#构建决策树桩
def buildStump(data, classlabel, D):
    m, n = data.shape
    numSteps = 10.0
    BestStump = {}  #储存最佳决策树桩参数
    bestClassEst = np.zeros(m)
    minError = np.inf
    for i in range(n):
        rangeMin = data[:, i].min()
        rangeMax = data[:, i].max()
        stepsize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshval = rangeMin + j * stepsize
                predictvals = stumpClassify(data, i, threshval, inequal)
                errArr = np.ones(m)
                errArr[predictvals == classlabel] = 0
                weightedError = D.dot(errArr)   #计算在D权重下的错误率

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictvals.copy()
                    BestStump['dim'] = i
                    BestStump['thresh'] = threshval
                    BestStump['ineq'] = inequal
    return BestStump, minError, bestClassEst


#基于决策树桩的AdaBoost训练
def adaBoostTrainDS(data, classlabels, numIt=40):
    weakClassArr = []
    m = data.shape[0]
    D = np.ones(m) / m
    aggClassEst = np.zeros(m)   #记录决策树桩累加结果
    for i in range(numIt):
        beststump, error, classEst = buildStump(data, classlabels, D)
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-16))   #计算系数alpha并方式除零溢出
        beststump['alpha'] = alpha
        weakClassArr.append(beststump)
        D = D * np.exp(-alpha * classlabels * classEst)     #更新权值
        D /= np.sum(D)    #归一化
        aggClassEst += alpha * classEst
        aggErrors = [1 if np.sign(aggClassEst[j]) != classlabels[j] else 0 for j in range(m)]
        errorrate = np.sum(aggErrors) / m
        if errorrate == 0:
            break
    return weakClassArr

#决策
def adaClassify(data, classifierArr):
    m = data.shape[0]
    aggClassEst = np.zeros(m)
    for i in range(len(classifierArr)):
        classEst = stumpClassify(data, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)

#测试
X = np.array([
    [1., 2.1],
    [2., 1.1],
    [1.3, 1.],
    [1., 1.],
    [2., 1.]
])
y = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
ada = adaBoostTrainDS(X, y, 30)
print(adaClassify(np.array([[0, 0], [5, 5]]), ada))

