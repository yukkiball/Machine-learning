import numpy as np

#定义树结点
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.parent = parentNode
        self.children = {}
        self.nodeLink = None    #链表指针

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(" " * ind, self.name, " ", self.count)
        for child in self.children.values():
            self.disp(ind+1)

#构建FP树
def createTree(dataSet, minSup=1):
    headerTable = {}    #头指针表
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    #删除低频项
    delete = [k for k in headerTable.keys() if headerTable[k] < minSup]
    for k in delete:
        del headerTable[k]

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0 :
        return None, None
    #头指针表（计数， 下一项）
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    retTree = treeNode('Null set', 1, None)     #根结点
    for tranSet, count in dataSet.items():
        localD = {} #记录全局频率
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            #按出现频率排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:    #链表为空
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink    #找到链表尾部
    nodeToTest.nodeLink = targetNode

#从FP树挖掘频繁集
def ascendTree(leafNode, prefixPath):
    #迭代回溯整颗树(dfs）
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

#找到所有前缀路径
def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

#查找频繁集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[0])]

    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)

        if myHead != None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


#测试
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

initSet = createInitSet(loadSimpDat())
myFPtree, myHeaderTab = createTree(initSet, 3)
freqItems = []
mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
print(freqItems)