"""
irisのボトムアップ型クラスタリング
--------------------
コメントアウトでクラスタリング手法を切り替える

def method(a, b):
    return single_linkage(a, b)
    # return complete_linkage(a, b)
    # return centroid_method(a.centroid, b.centroid)
    # return upgma(a, b)
    # return ward(a, b)
--------------------
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
from sklearn.decomposition import PCA
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram
import math

colors = ['r', 'g', 'b', 'c', 'm', 'y']
color_num = len(colors)

# PCA 4d -> 2d
iris4d = datasets.load_iris().data
pca = PCA(n_components=2)
iris2d = pca.fit_transform(iris4d)

fig = plt.scatter(iris2d[0:50, 0], iris2d[0:50, 1], s=20, marker='o', c="r", edgecolors="r", alpha=0.5)
fig = plt.scatter(iris2d[50:100, 0], iris2d[50:100, 1], s=20, marker='o', c="g", edgecolors="g", alpha=0.5)
fig = plt.scatter(iris2d[100:150, 0], iris2d[100:150, 1], s=20, marker='o', c="b", edgecolors="b", alpha=0.5)
plt.savefig("iris_pca.png", bbox_inches='tight', pad_inches=0.1)
plt.clf()

MAX_DATA = len(iris2d)

Z = []
nodeList= []
nodeMax = 0
class node:
    def __init__(self):
        self.node_num = -1
        self.datalist = []
        self.centroid = [0,0]
    def len(self):
        return len(self.datalist)

for i in range(len(iris2d)):
    tmp = node()
    tmp.node_num = nodeMax
    nodeMax += 1
    tmp.datalist.append(iris2d[i])
    tmp.centroid = iris2d[i]
    nodeList.append(tmp)

# ユークリッド距離の計算
def calcEuclid(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
# 単連結法
def single_linkage(a,b):
    minDist = np.inf
    for i in a.datalist:
        for j in b.datalist:
            dist = calcEuclid(i, j)
            if dist < minDist:
                minDist = dist
    return minDist
# 完全連結法
def complete_linkage(a,b):
    maxDist = -1
    for i in a.datalist:
        for j in b.datalist:
            dist = calcEuclid(i, j)
            if dist > maxDist:
                maxDist = dist
    return maxDist
# 重心法
def centroid_method(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
# 群平均法
def upgma(a, b):
    distSum = 0
    for i in a.datalist:
        for j in b.datalist:
            distSum += calcEuclid(i, j)
    return distSum / (a.len()*b.len())
# ウォード法
def ward(a, b):
    dist = a.len()*b.len() / (a.len()+b.len()) * calcEuclid(a.centroid, b.centroid)**2
    return math.sqrt(dist)
# メソッド切り替え
def method(a, b):
    return single_linkage(a, b)
    # return complete_linkage(a, b)
    # return centroid_method(a.centroid, b.centroid)
    # return upgma(a, b)
    # return ward(a, b)

for k in range(MAX_DATA-1):
    distMat = np.full((MAX_DATA-k, MAX_DATA-k), np.inf)
    for i in range(MAX_DATA-1-k):
        for j in range(i+1, MAX_DATA-k):
            distMat[i, j] = method(nodeList[i], nodeList[j])
    minIndex = np.unravel_index(np.argmin(distMat), distMat.shape)
    newNode = node()
    newNode.node_num = nodeMax
    nodeMax += 1
    newNode.datalist.extend(nodeList[minIndex[0]].datalist)
    newNode.datalist.extend(nodeList[minIndex[1]].datalist)
    newNode.centroid = np.mean(newNode.datalist, axis=0)
    zi = []
    zi += [nodeList[minIndex[0]].node_num, nodeList[minIndex[1]].node_num, np.min(distMat), newNode.len()]
    Z.append(zi)

    # plot data
    for i in range(len(nodeList)):
        for j in range(nodeList[i].len()):
            fig = plt.scatter(nodeList[i].datalist[j][0], nodeList[i].datalist[j][1], s=20, marker='o', c=colors[i%color_num], edgecolors=colors[i%color_num], alpha=0.5)
        fig = plt.scatter(nodeList[i].centroid[0], nodeList[i].centroid[1], marker='+', c='k', edgecolors='k')
    plt.savefig("iris{:03d}.png".format(k), bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    # //plot data

    nodeList.pop(minIndex[1])
    nodeList.pop(minIndex[0])
    nodeList.append(newNode)
# plot data
for i in range(len(nodeList)):
    for j in range(nodeList[i].len()):
        fig = plt.scatter(nodeList[i].datalist[j][0], nodeList[i].datalist[j][1], s=20, marker='o', c=colors[i%color_num], edgecolors=colors[i%color_num], alpha=0.5)
    fig = plt.scatter(nodeList[i].centroid[0], nodeList[i].centroid[1], marker='+', c='k', edgecolors='k')
plt.savefig("iris{:03d}.png".format(k+1), bbox_inches='tight', pad_inches=0.1)
plt.clf()
# //plot data

Z = np.array(Z)
dendrogram(Z)
plt.savefig("dendrogram.png", bbox_inches='tight', pad_inches=0.1)
plt.clf()
