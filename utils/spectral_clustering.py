# encoding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize


def similarity_function(points):
    """
    相似性函数，利用径向基核函数计算相似性矩阵，对角线元素置为0
    对角线元素为什么要置为０我也不清楚，但是论文里是这么说的
    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res


def spectral_clustering(points, k):
    """
    谱聚类
    :param points: 样本点
    :param k: 聚类个数
    :return: 聚类结果
    """
    W = similarity_function(points)
    # 度矩阵D可以从相似度矩阵W得到，这里计算的是D^(-1/2)
    # D = np.diag(np.sum(W, axis=1))
    # Dn = np.sqrt(LA.inv(D))
    # 本来应该像上面那样写，我做了点数学变换，写成了下面一行
    Dn = np.diag(np.power(np.sum(W, axis=1), -0.5))
    # 拉普拉斯矩阵：L=Dn*(D-W)*Dn=I-Dn*W*Dn
    # 也是做了数学变换的，简写为下面一行
    L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
    eigvals, eigvecs = LA.eig(L)
    # 前k小的特征值对应的索引，argsort函数
    indices = np.argsort(eigvals)[:k]
    # 取出前k小的特征值对应的特征向量，并进行正则化
    k_smallest_eigenvectors = normalize(eigvecs[:, indices])
    # 利用KMeans进行聚类
    return KMeans(n_clusters=k).fit_predict(k_smallest_eigenvectors)


if __name__ == "__main__":
    X, y = make_blobs(n_samples=150)
    # print(X)
    print(y)
    labels = spectral_clustering(X, 3)
    print(labels)

    # 画图
    plt.style.use('ggplot')
    # 原数据
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    ax0.scatter(X[:, 0], X[:, 1], c=y)
    ax0.set_title('raw data')
    # 谱聚类结果
    ax1.scatter(X[:, 0], X[:, 1], c=labels)
    ax1.set_title('Spectral Clustering')
    plt.show()