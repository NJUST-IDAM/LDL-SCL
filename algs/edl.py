from scipy.optimize import fmin_l_bfgs_b, fmin
import scipy.io as sio
from sklearn.model_selection import train_test_split
from numpy.matlib import repmat
from utils.evaluation_metrics import *


# read data set .mat
def read_mat(url):
    data = sio.loadmat(url)
    return data


if __name__ == "__main__":
    MAX_ITER = 100
    data1 = read_mat(r"../datasets/Human_Gene.mat")
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
    result7 = []
    result8 = []
    result9 = []
    result10 = []
    result11 = []
    for t in range(5):
        x_train, x_test, y_train, y_test = train_test_split(features, label_real, test_size=0.2, random_state=t)
        pear = np.corrcoef(y_train, rowvar=0)
        # for i in range(len(pear)):
        #     for j in range(len(pear[0])):
        #         if np.abs(pear[i][j]) < 0.:
        #             pear[i][j] = 0
        pear = pear / np.sum(pear, 1).reshape(-1, 1)
        xi1 = 0.5
        xi2 = 0.5

        def predict_func(x, m_theta):
            m_theta = m_theta.reshape(features_dim, labels_dim)
            numerator = np.exp(np.dot(x, m_theta))
            denominator = np.sum(np.exp(np.dot(x, m_theta)), 1).reshape(-1, 1)
            return numerator / denominator

        def obj_func(m_theta):
            m_theta = m_theta.reshape(features_dim, labels_dim)
            label_pre = predict_func(x_train, m_theta)
            cost1 = 0
            for i in range(len(x_train)):
                m1 = repmat(label_real[i], labels_dim, 1)
                m2 = repmat(label_pre[i].reshape(-1, 1), 1, labels_dim)
                cost1 += np.sum(pear * (m1-m2) * (np.log((m1+0.00001)/(m2+0.00001))))
            cost2 = np.sum((m_theta-np.mean(m_theta)) ** 2)
            cost3 = np.sum(m_theta ** 2)
            return cost1 - xi1*(1/len(x_train))*cost2 + 0.5*xi2*cost3

        init_theta = np.ones([features_dim, labels_dim])-np.random.rand(features_dim, labels_dim)/100
        result = fmin(obj_func, init_theta, maxiter=MAX_ITER)
        # print(result)
        label_pre = predict_func(x_test, result)
        # result1.append(euclidean(y_test, y_pre))
    #     print("No." + str(t) + ": " + str(euclidean(y_test, y_pre)))
    #     result2.append(sorensen(y_test, y_pre))
    #     print("No." + str(t) + ": " + str(sorensen(y_test, y_pre)))
    #     result3.append(squared_chi2(y_test, y_pre))
    #     print("No." + str(t) + ": " + str(squared_chi2(y_test, y_pre)))
    #     result4.append(kl(y_test+0.00001, y_pre+0.00001))
    #     print("No." + str(t) + ": " + str(kl(y_test+0.00001, y_pre+0.00001)))
    #     result5.append(intersection(y_test, y_pre))
    #     print("No." + str(t) + ": " + str(intersection(y_test, y_pre)))
    #     result6.append(fidelity(y_test, y_pre))
    #     print("No." + str(t) + ": " + str(fidelity(y_test, y_pre)))
    #
    # print("euclidean:", np.mean(result1), "+", np.std(result1))
    # print("sorensen:", np.mean(result2), "+", np.std(result2))
    # print("squared_chi2:", np.mean(result3), "+", np.std(result3))
    # print("kl:", np.mean(result4), "+", np.std(result4))
    # print("intersection:", np.mean(result5), "+", np.std(result5))
    # print("fidelity:", np.mean(result6), "+", np.std(result6))


        # SLDL six measures

        print(clark(y_test + 10 ** -6, label_pre + 10 ** -6))
        result1.append(clark(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(canberra(y_test + 10 ** -6, label_pre + 10 ** -6))
        result2.append(canberra(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(kl(y_test + 10 ** -6, label_pre + 10 ** -6))
        result3.append(kl(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(chebyshev(y_test + 10 ** -6, label_pre + 10 ** -6))
        result4.append(chebyshev(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(intersection(y_test + 10 ** -6, label_pre + 10 ** -6))
        result5.append(intersection(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(cosine(y_test + 10 ** -6, label_pre + 10 ** -6))
        result6.append(cosine(y_test + 10 ** -6, label_pre + 10 ** -6))

        # other measures

        print(euclidean(y_test + 10 ** -6, label_pre + 10 ** -6))
        result7.append(euclidean(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(squared_chi2(y_test + 10 ** -6, label_pre + 10 ** -6))
        result8.append(squared_chi2(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(fidelity(y_test + 10 ** -6, label_pre + 10 ** -6))
        result9.append(fidelity(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(sorensen(y_test + 10 ** -6, label_pre + 10 ** -6))
        result10.append(sorensen(y_test + 10 ** -6, label_pre + 10 ** -6))

        print(squared_chord(y_test + 10 ** -6, label_pre + 10 ** -6))
        result11.append(squared_chord(y_test + 10 ** -6, label_pre + 10 ** -6))


    print('\n\n')

    mea = [np.mean(result1), np.mean(result2), np.mean(result3), np.mean(result4), np.mean(result5), np.mean(result6)]
    stda = [np.std(result1), np.std(result2), np.std(result3), np.std(result4), np.std(result5), np.std(result6)]
    print(
        "      clark           canberra                kl                chebyshev            intersection                cosine        ")
    print(mea)
    print(stda)

    mea2 = [np.mean(result7), np.mean(result8), np.mean(result9), np.mean(result10), np.mean(result11)]
    stda2 = [np.std(result7), np.std(result8), np.std(result9), np.std(result10), np.std(result11)]
    print(
        "       euclidean           squared-chi2            fidelity            sorensen              squared-chord        ")
    print(mea2)
    print(stda2)







