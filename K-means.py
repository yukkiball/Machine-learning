import numpy as np
import matplotlib.pyplot as plt


#欧氏距离
def distance(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

#构建簇质心
def randCent(data, k):
    n = data.shape[1]
    centroids = np.zeros((k, n))
    for j in range(n):
        minJ = min(data[:, j])
        rangeJ = max(data[:, j]) - minJ
        centroids[:, j] = minJ + rangeJ * np.random.rand(k)
    return centroids

#K均值聚类
def Kmeans(data, k, dis=distance, createcent=randCent):
    m = data.shape[0]
    clusterassment = np.zeros((m, 2))
    centroids = createcent(data, k)
    clusterchanged = True
    while clusterchanged:
        clusterchanged = False
        for i in range(m):  #外循环（对每个样本）
            mindist = np.inf
            minindex = -1
            for j in range(k):  #内循环（对每个类）
                dist = dis(data[i], centroids[j])
                if dist < mindist:
                    mindist = dist
                    minindex = j
            if clusterassment[i, 0] != minindex:    #判断是否正确分类
                clusterchanged = True
            clusterassment[i, 0] = minindex
            clusterassment[i, 1] = mindist ** 2

        for cent in range(k):   #更新聚类中心
            pts = data[np.nonzero(clusterassment[:, 0] == cent)]    #在第cent类的样本
            centroids[cent] = np.mean(pts, axis=0)
    return centroids, clusterassment

#二分K均值聚类（避免局部最优）
def biKmeans(data, k, dis=distance):
    m = data.shape[0]
    clusterassment = np.zeros((m, 2))
    centroid0 = np.mean(data, axis=0)   #构建初始簇
    cenList = [centroid0]
    for j in range(m):
        clusterassment[j, 1] = dis(data[j], centroid0) ** 2
    while len(cenList) < k:
        lowestSSE = np.inf
        for i in range(len(cenList)):
            pts = data[np.nonzero(clusterassment[:, 0] == i)]   #当前簇的样本
            centroid, splitass = Kmeans(pts, 2, dis)
            sseSplit = np.sum(splitass[:, 1])
            ssenotSplit = np.sum(clusterassment[np.nonzero(clusterassment[:, 0] != i), 1])
            if sseSplit + ssenotSplit < lowestSSE:
                bestcenttosplit = i #确定要再次划分的簇
                bestnewcents = centroid
                bestclusterass = splitass.copy()
                lowestSSE = sseSplit + ssenotSplit
        bestclusterass[np.nonzero(bestclusterass[:, 0] == 1), 0] = len(cenList)     #划分的新簇的后半部分增添到列表尾部
        bestclusterass[np.nonzero(bestclusterass[:, 0] == 0), 0] = bestcenttosplit  #划分的新簇的前半部分替换原位置
        cenList[bestcenttosplit] = bestnewcents[0]
        cenList.append(bestnewcents[1])
        clusterassment[np.nonzero(clusterassment[:, 0] == bestcenttosplit)] = bestclusterass
    return np.array(cenList), clusterassment

#测试
X = np.array(
[
    [20, 20],
    [20, 21],
    [-20, 20],
    [-20, 21],
    [-20, -20],
    [-20, -21],
    [20, -20],
    [20, -21],
])
cenlist, assment = biKmeans(X, 4)
print(assment)
plt.scatter(X[:, 0], X[:, 1])
plt.show()




