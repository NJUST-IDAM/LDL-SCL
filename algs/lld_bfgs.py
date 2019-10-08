from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
import scipy.io as sio
from sklearn.model_selection import train_test_split
from utils.evaluation_metrics import *


# read data set .mat
def read_mat(url):
    data = sio.loadmat(url)
    return data


# x: single instance
def predict_func(x, m_theta, f_dim, l_dim):
    m_theta = m_theta.reshape(f_dim, l_dim)
    numerator = np.exp(np.dot(x, m_theta))
    denominator = np.sum(np.exp(np.dot(x, m_theta)), 1).reshape(-1, 1)
    return numerator / denominator


# x_train: all instances
def obj_func(m_theta, x_train, dis_train, f_dim, l_dim):
    m_theta = m_theta.reshape(f_dim, l_dim)
    dis_pre = predict_func(x_train, m_theta, f_dim, l_dim)
    return np.sum(dis_train * np.log((dis_train+0.00001) / (dis_pre+0.00001)))


# x_train: all instances
def fprime(m_theta, x_train, dis_train, f_dim, l_dim):
    m_theta = m_theta.reshape(f_dim, l_dim)
    modProb = np.exp(np.dot(x_train, m_theta))
    sumProb = np.sum(modProb, 1)
    modProb = modProb / sumProb.reshape(-1, 1)
    gradient = np.dot(np.transpose(x_train), (modProb - dis_train))
    return gradient.reshape(1, -1)[0]


if __name__ == "__main__":
    MAX_ITER = 70

    data1 = read_mat(r"../datasets/SBU_3DFE.mat")
    features = data1["features"]
    label_real = data1["labels"]
    features_dim = len(features[0])
    labels_dim = len(label_real[0])

    result1 = []
    result2 = []
    result3 = []
    result4 = []
    result5 = []
    result6 = []

    for t in range(10):
        x_train, x_test, y_train, y_test = train_test_split(features, label_real, test_size=0.2, random_state=t)
        # init_theta = np.ones([features_dim, labels_dim])
        init_theta = np.random.rand(features_dim, labels_dim)
        result = fmin_bfgs(obj_func, init_theta, fprime, args=(x_train, y_train, features_dim, labels_dim),
                           maxiter=MAX_ITER, disp=True)

        pre_test = predict_func(x_test, result, features_dim, labels_dim)

        result1.append(euclidean(y_test, pre_test))
        print("No." + str(t) + ": " + str(euclidean(y_test, pre_test)))
        result2.append(sorensen(y_test, pre_test))
        print("No." + str(t) + ": " + str(sorensen(y_test, pre_test)))
        result3.append(squared_chi2(y_test, pre_test))
        print("No." + str(t) + ": " + str(squared_chi2(y_test, pre_test)))
        result4.append(kl(y_test + 0.00001, pre_test + 0.00001))
        print("No." + str(t) + ": " + str(kl(y_test + 0.00001, pre_test + 0.00001)))
        result5.append(intersection(y_test, pre_test))
        print("No." + str(t) + ": " + str(intersection(y_test, pre_test)))
        result6.append(fidelity(y_test, pre_test))
        print("No." + str(t) + ": " + str(fidelity(y_test, pre_test)))

    print("euclidean:", np.mean(result1), "+", np.std(result1))
    print("sorensen:", np.mean(result2), "+", np.std(result2))
    print("squared_chi2:", np.mean(result3), "+", np.std(result3))
    print("kl:", np.mean(result4), "+", np.std(result4))
    print("intersection:", np.mean(result5), "+", np.std(result5))
    print("fidelity:", np.mean(result6), "+", np.std(result6))

