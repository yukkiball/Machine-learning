import numpy as np


#构建只有一个元素的集合
def createC1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    C = []
    for i in C1:
        C.append(frozenset(i))
    return C

#计算所有项的支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    #计算每一项出现的频数
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.__contains__(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportdata = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportdata[key] = support
    return retList, supportdata

#使用Apriori算法寻找频繁集

#将两个k-1集合合并为k集合
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataset, minSuport=0.9):
    C1 = createC1(dataset)
    D = []
    for d in dataset:
        D.append(set(d))
    L1, supportdata = scanD(D, C1, minSuport)
    L = [L1]    #频繁集
    k = 2   #频繁集元素个数
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSuport)
        supportdata.update(supK)
        L.append(Lk)
        k += 1
    return L, supportdata

#建立关联规则
def generateRules(L, supportData, minConf=0.7):
    bigRulelist = []
    for i in range(1, len(L)):  #针对元素个数大于1的列表建立规则
        for freqset in L[i]:
            H1 = [frozenset([item]) for item in freqset]
            if i > 1:
                rulesFromConseq(freqset, H1, supportData, bigRulelist, minConf)
            else:
                calcConf(freqset, H1, supportData, bigRulelist, minConf)
    return bigRulelist

def calcConf(freqset, H, supportdata, brl, minConf=0.7):
    prunedH = []    #储存所有可能在规则右边的值
    for conseq in H:
        conf = supportdata[freqset] / supportdata[freqset - conseq]
        if conf >= minConf:
            prunedH.append(conseq)
            brl.append((freqset - conseq, conseq, conf))
    return prunedH

def rulesFromConseq(freqset, H, supportData, brl, minConf=0.7):
    m = len(H[0])   #待合并的子集大小
    if m == 1:
        calcConf(freqset, H, supportData, brl, minConf)
    if len(freqset) > m + 1:
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqset, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqset, Hmp1, supportData, brl, minConf)

#测试
dataset = [
    [1, 3, 4],
    [2, 3, 5],
    [1, 2, 3, 5],
    [2, 5]
]
L, suppdata = apriori(dataset, minSuport=0.5)
rule = generateRules(L, suppdata, minConf=0.5)
print(rule)
