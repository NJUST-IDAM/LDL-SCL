import numpy as np


def euclidean(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(np.sum((distribution_real - distribution_predict) ** 2, 1))) / height


def squared_chord(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = (np.sqrt(distribution_real) - np.sqrt(distribution_predict)) ** 2
    denominator = np.sum(numerator)
    return denominator / height

def sorensen(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = np.sum(np.abs(distribution_real - distribution_predict), 1)
    denominator = np.sum(distribution_real + distribution_predict, 1)
    return np.sum(numerator / denominator) / height

def squared_chi2(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = (distribution_real - distribution_predict) ** 2
    denominator = distribution_real + distribution_predict
    return np.sum(numerator / denominator) / height


def kl(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(distribution_real * np.log(distribution_real / distribution_predict)) / height


def intersection(distribution_real, distribution_predict):
    height, width = distribution_real.shape
    inter = 0.
    for i in range(height):
        for j in range(width):
            inter += np.min([distribution_real[i][j], distribution_predict[i][j]])
    return inter / height


def fidelity(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(distribution_real * distribution_predict)) / height


def chebyshev(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.max(np.abs(distribution_real-distribution_predict), 1)) / height


def clark(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(np.sum((distribution_real-distribution_predict)**2 / (distribution_real+distribution_predict)**2, 1))) / height


def canberra(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.abs(distribution_real-distribution_predict) / (distribution_real+distribution_predict)) / height


def cosine(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sum(distribution_real*distribution_predict, 1) / (np.sqrt(np.sum(distribution_real**2, 1)) *\
           np.sqrt(np.sum(distribution_predict**2, 1)))) / height


if __name__ == "__main__":
    real = np.array([[0.5, 0.5], [0.5, 0.5]])
    predict = np.array([[0.4, 0.6], [0.7, 0.3]])
    print(euclidean(real, predict))
    print(sorensen(real, predict))
    print(squared_chi2(real, predict))
    print(kl(real, predict))
    print(intersection(real, predict))
    print(fidelity(real, predict))
    print(chebyshev(real, predict))
    print(clark(real, predict))
    print(canberra(real, predict))
    print(cosine(real, predict))


