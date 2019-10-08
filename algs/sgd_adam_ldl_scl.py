# -*- coding:utf-8 -*-
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils.evaluation_metrics import *
from utils.spectral_clustering import spectral_clustering
import matplotlib.pyplot as plt

f = open("log.txt", "w")


# read data set .mat
def read_mat(url):
    data = sio.loadmat(url)
    return data


# next batch
def next_batch(num, data, labels, codes):
    # Return a total of `num` random samples and labels.
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    codes_shuffle = [codes[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(codes_shuffle)


# cluster
def cluster_ave(labels_train, n):
    train_len = len(labels_train)
    # using k-means
    kmeans = KMeans(n_clusters=n, random_state=0).fit(labels_train)
    predict = kmeans.predict(labels_train)
    # using spectral clustering
    # predict = spectral_clustering(labels_train, n)
    classification = []
    for i in range(n):
        classification.append([])
    c = np.zeros([train_len, n]) + 10 ** -6
    for i in range(train_len):
        c[i][predict[i]] = 1
        classification[predict[i]].append(labels_train[i])
    p = []
    for i in range(n):
        p.append(np.average(classification[i], 0))
    p = np.array(p)
    return c, p


# x: matrix of feature, n * d
# theta: weight matrix of feature, d * l, l is the number of labels
# c: matrix of code, n * m, m is the number of clusters
# w: weight matrix of code matrix, m * l
def predict_func(x, theta, c, w):
    matrix = np.dot(x, theta) + np.dot(c, w)
    matrix1 = matrix - np.max(matrix, 1).reshape(-1, 1)
    numerator = np.exp(matrix1)
    denominator = np.sum(np.exp(matrix1), 1).reshape(-1, 1)
    return numerator / denominator


# label_real: real label of instance, n * l
# p: the average vector of cluster, number of clusters * l
def optimize_func(x, theta, c, w, label_real, p, lambda1, lambda2, lambda3, mu):
    label_predict = predict_func(x, theta, c, w) + 10 ** -6
    term1 = np.sum(label_real * np.log((label_real + 10 ** -6) / (label_predict + 10 ** -6)))
    term2 = np.sum(theta ** 2)
    term3 = np.sum(w ** 2)
    dist = []
    for i in range(len(p)):
        dist.append(np.sum((label_predict - p[i]) ** 2, 1))
    dist = np.array(dist).T
    term4 = np.sum(c * dist)
    term5 = np.sum(1. / c)
    return term1 + lambda1 * term2 + lambda2 * term3 + lambda3 * term4 + mu * term5


# m: the row of theta
# n: the column of theta
# def gradient_theta(x, theta, c, w, label_real, p, m, n, lambda1, lambda2, lambda3):
#     # the first term
#     gradient1 = np.sum((predict_func(x, theta, c, w)[:, n] - label_real[:, n]) * x[:, m])
#     # the second term
#     gradient2 = 2 * lambda1 * theta[m][n]
#     # the third term
#     gradient3 = 0.
#     for i in range(len(x)):
#         for j in range(len(p)):
#             denominator = np.sum(np.exp(np.dot(x[i], theta) + np.dot(c[i], w)))
#             p_i_n = np.exp(np.dot(x[i], theta) + np.dot(c[i], w))[n] / denominator
#             gradient3 += c[i][j] * (p_i_n-p[j][n]) * x[i][m] * (p_i_n-p_i_n**2)
#     gradient3 *= 2*lambda3
#     return gradient1 + gradient2 + gradient3
#
#
# def gradient_w(x, theta, c, w, label_real, p, m, n, lambda1, lambda2, lambda3):
#     # the first term
#     gradient1 = np.sum((predict_func(x, theta, c, w)[:, n] - label_real[:, n]) * c[:, m])
#     # the second term
#     gradient2 = 2 * lambda2 * w[m][n]
#     # the third term
#     gradient3 = 0.
#     for i in range(len(x)):
#         for j in range(len(p)):
#             denominator = np.sum(np.exp(np.dot(x[i], theta) + np.dot(c[i], w)))
#             p_i_n = np.exp(np.dot(x[i], theta) + np.dot(c[i], w))[n] / denominator
#             gradient3 += c[i][j] * (p_i_n-p[j][n]) * c[i][m] * (p_i_n-p_i_n**2)
#     gradient3 *= 2*lambda3
#     return gradient1 + gradient2 + gradient3
#
#
# def gradient_c(x, theta, c, w, label_real, p, m, n, lambda1, lambda2, lambda3, mu):
#     # the first term
#     gradient1 = -np.sum(label_real[m] * w[n])
#     # the second term
#     numerator = np.sum(np.exp(np.dot(x[m], theta) + np.dot(c[m], w)) * w[n])
#     denominator = np.sum(np.exp(np.dot(x[m], theta) + np.dot(c[m], w)))
#     gradient2 = numerator / denominator
#     # the third term
#     gradient3 = 0.
#     for l in range(len(label_real[0])):
#         denominator = np.sum(np.exp(np.dot(x[m], theta) + np.dot(c[m], w)))
#         p_i_l = np.exp(np.dot(x[m], theta) + np.dot(c[m], w))[l] / denominator
#         numerator1 = np.sum(np.exp(np.dot(x[m], theta) + np.dot(c[m], w)) * w[n])
#         partial_c = p_i_l * (w[n][l] - numerator1/denominator)
#         gradient3 += (p_i_l - p[n][l]) * partial_c
#     gradient3 *= 2 * lambda3 * c[m][n]
#     # the fourth term
#     denominator = np.sum(np.exp(np.dot(x[m], theta) + np.dot(c[m], w)))
#     p_i = np.exp(np.dot(x[m], theta) + np.dot(c[m], w)) / denominator
#     gradient4 = lambda3 * np.sum((p_i - p[n]) ** 2)
#     # the fifth term
#     gradient5 = -mu * c[m][n] ** (-2)
#     return gradient1 + gradient2 + gradient3 + gradient4 + gradient5



def gradient_theta(x, theta, c, w, label_real, P, m, n, lambda1, lambda2, lambda3):
    gradient1 = x.T.dot(predict_func(x, theta, c, w) - label_real)
    gradient2 = 2 * lambda1 * theta
    p_tmp = np.exp(np.dot(x, theta) + np.dot(c, w))
    p = (p_tmp.T / np.sum(p_tmp, axis=1)).T
    gradient3 = x.T.dot(np.multiply((len(P)*c.sum(axis=1)*p.T).T - c.dot(P), p-p**2))
    gradient3 *= 2*lambda3
    return gradient1 + gradient2 + gradient3

# @exeTime
def gradient_w(x, theta, c, w, label_real, P, m, n, lambda1, lambda2, lambda3):
    gradient1 = c.T.dot(predict_func(x, theta, c, w) - label_real)
    gradient2 = 2 * lambda2 * w
    p_tmp = np.exp(np.dot(x, theta) + np.dot(c, w))
    p = (p_tmp.T / np.sum(p_tmp, axis=1)).T
    gradient3 = c.T.dot(np.multiply((len(P)*c.sum(axis=1)*p.T).T - c.dot(P), p-p**2))
    gradient3 *= 2*lambda3
    return gradient1 + gradient2 + gradient3

# @exeTime
def gradient_c(x, theta, c, w, label_real, P, m, n, lambda1, lambda2, lambda3, mu):
    gradient1 = -label_real.dot(w.T)
    p_tmp = np.exp(np.dot(x, theta) + np.dot(c, w))
    p = (p_tmp.T / np.sum(p_tmp, axis=1)).T
    numerator = p_tmp.dot(w.T)
    denominator = np.sum(p_tmp, axis=1)
    gradient2 = (numerator.T/denominator).T

    gradient3 = np.zeros((len(x_train), code_len))
    for m in range(len(x_train)):
        for n in range(code_len):
            grad = 0.
            for l in range(len(label_real[0])):
                grad += (p[m][l] - P[n][l]) * p[m][l] * (w[n][l] - gradient2[m][n])
            grad *= 2 * lambda3 * c[m][n]
            gradient3[m][n] = grad

    a = np.sum(p * p, 1)
    b = np.sum(P * P, 1)
    ab = p.dot(P.T)
    gradient4 = lambda3 * np.abs(np.repeat(a.reshape(-1, 1), len(P), 1) + np.repeat(np.array([b]), len(p), 0) - 2 * ab)
    gradient5 = - mu * c ** (-2)
    return gradient1 + gradient2 + gradient3 + gradient4 + gradient5





if __name__ == "__main__":
    # configuration
    lambda1 = 0.001
    lambda2 = 0.001
    lambda3 = 0.1
    #code_len = 5
    iters = 300
    batch = 50

    rho1 = 0.9
    rho2 = 0.999
    delta = 10 ** -8    # smoothing term
    epsilon = 0.001     # learning rate
    for c in range(15):
        code_len = c+1
        print("-"*50+str(code_len)+'-'*50)

        data1 = read_mat(r"../datasets/Yeast_cold.mat")
        features = data1["features"]
        # features = data1["features"][:, 0:168]
        label_real1 = data1["labels"]
        features_dim = len(features[0])
        labels_dim = len(label_real1[0])

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

        '''result21 = []
        result22 = []
        result23 = []
        result24 = []
        result25 = []
        result26 = []
        result27 = []
        result28 = []
        result29 = []
        result30 = []
        result31 = []'''

        for t in range(10):
            #print('-'*50 + str(t) + '-'*50)
            s1 = r1 = np.zeros([features_dim, labels_dim])
            s2 = r2 = np.zeros([code_len, labels_dim])

            mu = 1
            theta1 = np.ones([features_dim, labels_dim])
            w1 = np.ones([code_len, labels_dim])

            x_train, x_test, y_train, y_test = train_test_split(features, label_real1, test_size=0.2, random_state=t)
            c1, p1 = cluster_ave(y_train, code_len)
            s3 = r3 = np.zeros_like(c1)

            loss1 = optimize_func(x_train, theta1, c1, w1, y_train, p1, lambda1, lambda2, lambda3, mu)

            # train starts
            loss = []
            #loss.append(loss1)
            for i in range(iters):
                x_batch, y_batch, c_batch = next_batch(batch, x_train, y_train, c1)

                gradient1 = gradient_theta(x_batch, theta1, c_batch, w1, y_batch, p1, 0, 0, lambda1, lambda2, lambda3)
                s1 = rho1 * s1 + (1 - rho1) * gradient1
                s1_hat = s1 / (1 - rho1 ** (i+1))
                r1 = rho2 * r1 + (1 - rho2) * gradient1 * gradient1
                r1_hat = r1 / (1 - rho2 ** (i+1))

                gradient2 = gradient_w(x_batch, theta1, c_batch, w1, y_batch, p1, 0, 0, lambda1, lambda2, lambda3)
                s2 = rho1 * s2 + (1 - rho1) * gradient2
                s2_hat = s2 / (1 - rho1 ** (i + 1))
                r2 = rho2 * r2 + (1 - rho2) * gradient2 * gradient2
                r2_hat = r2 / (1 - rho2 ** (i + 1))

                gradient3 = gradient_c(x_train, theta1, c1, w1, y_train, p1, 0, 0, lambda1, lambda2, lambda3, mu)
                s3 = rho1 * s3 + (1 - rho1) * gradient3
                s3_hat = s3 / (1 - rho1 ** (i + 1))
                r3 = rho2 * r3 + (1 - rho2) * gradient3 * gradient3
                r3_hat = r3 / (1 - rho2 ** (i + 1))

                theta1 = theta1 - epsilon * s1_hat / (np.sqrt(r1_hat)+delta)
                w1 = w1 - epsilon * s2_hat / (np.sqrt(r2_hat)+delta)
                c1 = c1 - epsilon * s3_hat / (np.sqrt(r3_hat)+delta)
                # print(predict_func(x_train, theta1, c1, w1))
                # print(w1)
                # print(c1)
                # print(theta1)

                loss2 = optimize_func(x_train, theta1, c1, w1, y_train, p1, lambda1, lambda2, lambda3, mu)
                if i > 5:
                    loss.append(loss2)
                #print(loss2)

                # print(kl(label_real1, predict_func(x1, theta1, c1, w1)))
                # if np.abs(loss2 - loss1) < 0.001 or loss2 > loss1 or mu*np.sum(1. / c1) < 10 ** -9:
                if np.abs(loss2 - loss1) < 0.0001:
                    break
                else:
                    mu = mu * 0.1
                loss1 = loss2
                #print("*" * 50, i)
            #plt.plot(loss)
            #plt.show()

            # print(theta1)
            # print(w1)
            # print(c1)

            # test starts
            regression = []
            for i in range(code_len):
                lr = LinearRegression()
                lr.fit(x_train, c1[:, i].reshape(-1, 1))
                regression.append(lr)
            codes = []
            for i in range(len(x_test)):
                for lr1 in regression:
                    codes.append(lr1.predict(x_test[i].reshape(1, -1)))
            codes = np.array(codes).reshape(len(x_test), code_len)
            label_pre = predict_func(x_test, theta1, codes, w1)

            '''codes_train = []
            for i in range(len(x_train)):
                for lr1 in regression:
                    codes_train.append(lr1.predict(x_train[i].reshape(1, -1)))
            codes_train = np.array(codes_train).reshape(len(x_train), code_len)
            label_train_pre = predict_func(x_train, theta1, codes_train, w1)'''
            # print(label_pre)
            # print(y_test)


            # SLDL six measures


            #print(clark(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(clark(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result1.append(clark(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(canberra(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(canberra(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result2.append(canberra(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(kl(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(kl(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result3.append(kl(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(chebyshev(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(chebyshev(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result4.append(chebyshev(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(intersection(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(intersection(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result5.append(intersection(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(cosine(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(cosine(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result6.append(cosine(y_test + 10 ** -6, label_pre + 10 ** -6))



            # other measures

            #print(euclidean(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(euclidean(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result7.append(euclidean(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(squared_chi2(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(squared_chi2(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result8.append(squared_chi2(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(fidelity(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(fidelity(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result9.append(fidelity(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(sorensen(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(sorensen(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n')
            result10.append(sorensen(y_test + 10 ** -6, label_pre + 10 ** -6))

            #print(squared_chord(y_test + 10 ** -6, label_pre + 10 ** -6))
            f.write(str(squared_chord(y_test + 10 ** -6, label_pre + 10 ** -6))+'\n\n\n')
            result11.append(squared_chord(y_test + 10 ** -6, label_pre + 10 ** -6))

            '''print('train\n')
    
    
            print(clark(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result21.append(clark(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(canberra(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result22.append(canberra(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(kl(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result23.append(kl(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(chebyshev(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result24.append(chebyshev(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(intersection(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result25.append(intersection(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(cosine(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result26.append(cosine(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
    
    
            # other measures
    
            print(euclidean(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result27.append(euclidean(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(squared_chi2(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result28.append(squared_chi2(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(fidelity(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result29.append(fidelity(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(sorensen(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result30.append(sorensen(y_train + 10 ** -6, label_train_pre + 10 ** -6))
    
            print(squared_chord(y_train + 10 ** -6, label_train_pre + 10 ** -6))
            result31.append(squared_chord(y_train + 10 ** -6, label_train_pre + 10 ** -6))'''









        # print(result1)
        # print(result2)
        # print(result3)
        # print(result4)
        # print(result5)
        # print(result6)
        # print("*" * 50)
        # print("sorensen:", np.mean(result1))
        # print("sorensen:", np.std(result1))
        # print("-" * 50)
        # print("squared-chord:", np.mean(result2))
        # print("squared-chord:", np.std(result2))
        # print("-" * 50)
        # print("kl:", np.mean(result3))
        # print("kl:", np.std(result3))
        # print("-" * 50)
        # print("chebyshev:", np.mean(result4))
        # print("chebyshev:", np.std(result4))
        # print("-" * 50)
        # print("intersection:", np.mean(result5))
        # print("intersection:", np.std(result5))
        # print("-" * 50)
        # print("cosine:", np.mean(result6))
        # print("cosine:", np.std(result6))
        # print("-" * 50)
        # print("euclidean:", np.mean(result7))
        # print("euclidean:", np.std(result7))
        # print("-" * 50)
        # print("squared_chi2:", np.mean(result8))
        # print("squared_chi2:", np.std(result8))
        # print("-" * 50)
        # print("fidelity:", np.mean(result9))
        # print("fidelity:", np.std(result9))
        # print("-" * 50)
        # print("clark:", np.mean(result10))
        # print("clark:", np.std(result10))
        # print("-" * 50)
        # print("canberra:", np.mean(result11))
        # print("canberra:", np.std(result11))


        print('\n\n')

        mea = [np.mean(result1),np.mean(result2),np.mean(result3),np.mean(result4),np.mean(result5),np.mean(result6)]
        stda = [np.std(result1),np.std(result2),np.std(result3),np.std(result4),np.std(result5),np.std(result6)]
        print("      clark           canberra                kl                chebyshev            intersection                cosine        ")
        print(mea)
        print(stda)

        mea2 = [np.mean(result7),np.mean(result8),np.mean(result9),np.mean(result10),np.mean(result11)]
        stda2 = [np.std(result7),np.std(result8),np.std(result9),np.std(result10),np.std(result11)]
        print("       euclidean           squared-chi2            fidelity            sorensen              squared-chord        ")
        print(mea2)
        print(stda2)

        '''print('train\n\n')
    
        mea3 = [np.mean(result21), np.mean(result22), np.mean(result23), np.mean(result24), np.mean(result25), np.mean(result26)]
        stda3 = [np.std(result21), np.std(result22), np.std(result23), np.std(result24), np.std(result25), np.std(result26)]
        print(
            "      clark           canberra                kl                chebyshev            intersection                cosine        ")
        print(mea3)
        print(stda3)
    
        mea4 = [np.mean(result27), np.mean(result28), np.mean(result29), np.mean(result30), np.mean(result31)]
        stda4 = [np.std(result27), np.std(result28), np.std(result29), np.std(result30), np.std(result31)]
        print(
            "       euclidean           squared-chi2            fidelity            sorensen              squared-chord        ")
        print(mea4)
        print(stda4)'''

    f.close()
