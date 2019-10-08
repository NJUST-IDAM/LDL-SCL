# -*- coding: utf-8 -*-
"""
@paper:Clustering by fast search and find of density peak
@summary: 基于密度的聚类算法
    K-means是通过指定聚类中心，再通过迭代的方式更新聚类中心，每个点都被指派到距离最近的聚类中心，导致其不能检测非球面类别的数据分布。
    DBSCAN对于任意形状分布的进行聚类，但是必须指定一个密度阈值，从而去除低于此密度阈值的噪音点。
    这篇文章假设聚类中心周围都是密度比其低的点，同时这些点距离该聚类中心的距离相比于其他聚类中心最近。

@ImplementAuthor: Shaobo
Created on Mon Oct 20 13:38:18 2014
@ModifyAuthor: xiangzheng
changed on 2018/06/21
"""

from math import exp
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


MAX = 1000000000


def standardize(li):
    minV, maxV = min(li), max(li)
    return [float(item-minV)/(maxV-minV) for item in li]


def caculateDistance(points, norm=2):
    quantity = len(points)
    dist = np.zeros((quantity, quantity))
    for iTP in range(quantity - 1):
        for iOP in range(iTP + 1, quantity):
            dd = np.linalg.norm(points[iTP] - points[iOP], norm)
            dist[iTP][iOP] = dd
            dist[iOP][iTP] = dd
    return dist


def caculateDC(dist, percent=2.0):
    # percent : the average percentage of neighbours
    # return distance threshold value
    # maybe we can use linear selection algrothm instead of sorting
    ll = np.sort(dist, axis=None)[dist.shape[0]:]
    # print ll
    # if dc == 0 , enlarge percent till dc > 0
    dc = 0
    for i in range(1, int(100 / percent)):
        dc = ll[int(ll.shape[0] * percent * i / 100)]
        print('percent:', i * percent, 'dc:', dc)
        if dc > 0:
            break
    return dc


def caculateRho(dist, dc):
    # calculate local density of point
    quantity = dist.shape[0]
    rhos = np.zeros((quantity, 1))
    for iTP in range(quantity - 1):
        for iOP in range(iTP + 1, quantity):
            xxx = exp(-(dist[iTP][iOP] / dc) ** 2)
            # adding a random to distance is
            # to avoid multiple points have same rho
            rd = random.randint(0, 100000) / 100000.0 * xxx / 100000.0
            rhos[iTP] += (xxx + rd)
            rhos[iOP] += (xxx + rd)
    return rhos


def argDelta(iTP, dist, rhos):
    # 求离该点最近的,比该点密度大的点
    higherRhos = [[iOP, dist[iTP][iOP]]
                  for iOP in range(dist.shape[0])
                  if rhos[iOP] > rhos[iTP]]
    # print rhos
    # return min(higherRhos,key=lambda x: x[1])[0]
    higherRhos.sort(key=lambda x: x[1])
    return higherRhos[0][0]


def calcDeltas(dist, rhos):
    # 求比该点局部密度大的点到该点的最小距离
    # 如果这个点已经是局部密度最大的点，那么Delta赋值为别的所有点到它的最大距离。
    quantity = dist.shape[0]
    deltas = np.ones((quantity, 1)) * MAX
    maxDensity = np.max(rhos)
    for iTP in range(quantity):  # TP: this point,OP: other point
        if rhos[iTP] < maxDensity:
            deltas[iTP] = dist[iTP][argDelta(iTP, dist, rhos)]
        else:
            deltas[iTP] = max(dist[iTP])
    return deltas


def calcGammas(rhos, deltas):
    dd = deltas.copy()
    rr = rhos.copy()
    dd = np.array(standardize(dd))
    rr = np.array(standardize(rr))
    return dd * rr


def plotGammas(gammas):
    sortedGammas = np.sort(gammas, axis=0)[::-1]
    print('gammas:')
    for gamma in sortedGammas[: int(len(gammas) / 10)]:
        print('%.3f' % gamma,)
    print()
    plt.plot(range(len(gammas)), sortedGammas, '*')
    plt.xlabel('n'), plt.ylabel('gamma')
    plt.show()


def obtainCenters(gammas):
    # 确定聚类中心
    # print gammas
    sortIndex = np.argsort(gammas, axis=0)[::-1]
    classNum = int(input('how many clusters do you want?').strip())
    return sortIndex[:classNum]


def nearestNeighbor(iTP, dist, rhos, result):
    neighbor = argDelta(iTP, dist, rhos)
    if result[neighbor] == -1:
        result[neighbor] = nearestNeighbor(neighbor, dist, rhos, result)
    return result[neighbor]


def labelAllPoints(dist, rhos, centers):
    quantity = dist.shape[0]
    result = np.ones(quantity, dtype=np.int) * (-1)
    # 赋予中心点聚类类标
    for iCenter, center in enumerate(centers):
        result[center] = iCenter

    for i in range(quantity):
        dist[i][i] = MAX

    # 赋予每个点聚类类标
    rhos.shape = 1, -1
    argsortRhosR = np.argsort(rhos[0])[::-1]
    rhos.shape = -1, 1
    # argsortRhosR是按密度由大到小排序的下标,使用argsortRhosR和使用自然顺序下标有何区别?
    # range(quantity)
    for iTP in argsortRhosR:
        if result[iTP] == -1:
            result[iTP] = nearestNeighbor(iTP, dist, rhos, result)
    return result


def DPCA(dist, percent):
    dc = caculateDC(dist, percent)
    rhos = caculateRho(dist, dc)
    deltas = calcDeltas(dist, rhos)
    gammas = calcGammas(rhos, deltas)
    plotGammas(gammas)
    centers = obtainCenters(gammas)
    result = labelAllPoints(dist, rhos, centers)
    return result


if __name__ == '__main__':
    iris = load_iris()
    points = iris["data"][:, 2:4]
    labels = iris["target"]
    print('read done!')

    percent = 2.0
    norm = 2
    dist = caculateDistance(points, norm)
    result = DPCA(dist, percent)

    print(result)
    print(labels)

    # This can only plot 2D data. If you want to test your data, comment the following codes
    fig = plt.figure()
    plt.subplot(121)
    plt.title("real")
    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.subplot(122)
    plt.title("predicted")
    plt.scatter(points[:, 0], points[:, 1], c=result)
    plt.show()



