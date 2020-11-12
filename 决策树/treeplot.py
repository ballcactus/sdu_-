import matplotlib.pyplot as plt


#定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8") #定义判断节点形态
leafNode = dict(boxstyle="round4", fc="0.8") #定义叶节点形态
arrow_args = dict(arrowstyle="<-") #定义箭头


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def createPlot(inTree,classname):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree,classname)) #树的宽度
    plotTree.totalD = float(getTreeDepth(inTree,classname)) #树的深度
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '',classname)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 显示中文
    plt.show()

def plotTree(myTree, parentPt, nodeTxt,classname):
    numLeafs = getNumLeafs(myTree,classname)  #树叶节点数
    depth = getTreeDepth(myTree,classname)    #树的层数
    firstStr = list(myTree.keys())[0]     #节点标签
    #计算当前节点的位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/(2.0*plotTree.totalW), plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
     # 绘制带箭头的注解
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    if firstStr!= classname:
          # 在父子节点间填充文本信息
        plotNode(firstStr, cntrPt, parentPt, decisionNode)
        for key in secondDict.keys():
            plotTree(secondDict[key],cntrPt,str(key),classname)        #递归绘制树形图
    else:   #如果是叶节点
        plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
        plotNode(secondDict, cntrPt, parentPt, decisionNode)  # 绘制带箭头的注解
       # plotNode(secondDict, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)

    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


def getNumLeafs(myTree,classname):
    numLeafs = 0

    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    if type(secondDict).__name__=='dict':
        for key in secondDict.keys():
            numLeafs += getNumLeafs(secondDict[key],classname)  # 递归调用getNumLeafs
        else:
            numLeafs += 1  # 如果是叶节点，则叶节点+1

    return numLeafs


# 计算数的层数
def getTreeDepth(myTree,classname):
    maxDepth = 0
    thisDepth=0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    if type(secondDict).__name__ == 'dict':
        for key in secondDict.keys():
            thisDepth = 1 + getTreeDepth(secondDict[key],classname) #如果是字典，则层数加1，再递归调用getTreeDepth
            if thisDepth > maxDepth:
                maxDepth = thisDepth
    else:
        thisDepth = 1
        return thisDepth
        #得到最大层数
    return maxDepth