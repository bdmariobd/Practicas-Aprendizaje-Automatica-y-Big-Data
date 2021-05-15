#!/usr/bin/python3

# MIT License
#
# Copyright (c) 2021 Mario Blanco Dominguez, Juan Tecedor
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
import sklearn.preprocessing


def sigmoid(X):
    # x > 50 returns 1
    return 1 / (1 + np.exp(-X))


def costRegression(T, XX, Y, l):
    m = Y.size
    H = sigmoid(XX.dot(T))
    l1 = np.transpose(np.log(H))
    l2 = np.transpose(np.log(1 - H + 1e-6))

    ret = (-1 / m) * ((np.matmul(l1, Y)) + (np.matmul(l2, (1 - Y))))
    return ret + l / (2 * m) * np.sum(H * H)


def gradiente(theta, X, Y, l):
    m = np.shape(X)[0]
    H = sigmoid(X.dot(theta))
    ret = (1 / m) * np.matmul(np.transpose(X), H - Y)
    return ret + (l / m) * theta


def oneVsAll(X, Y, num_etiquetas, l):
    result = []
    mapFeature = sklearn.preprocessing.PolynomialFeatures(2)
    mapFeatureX = mapFeature.fit_transform(X)
    # mapFeatureX = X
    T = np.zeros(mapFeatureX.shape[1])

    for i in range(num_etiquetas):
        YY = (Y == i) * 1
        result.append(opt.fmin_tnc(func = costRegression, x0 = T, fprime = gradiente,
            args = (mapFeatureX, YY, l))[0])
    return result, mapFeatureX


def calcularAciertos(X, Y, T, tags):
    aciertos = 0
    j = 0
    print(X.shape)
    print(T.shape)
    for i in range(len(X)):
        for tag in range(tags):
            print(X[i].shape, T[tag].shape)
            print(X[i], T[tag])
            pred = sigmoid(X[i].dot(T[tag]))
            if pred < .5:
                pred = 0
            else:
                pred = 1
                input()

            pred = int(pred)
            print(pred, Y[j], tag)
            if pred == 1 and ((Y[j] == 10 and tag == 0) or tag == Y[j]):
                print('hola')
                aciertos += 1
                break
        j += 1
    return aciertos / len(Y) * 100


def main():
    data = loadmat('./ex3data1.mat')
    X = data['X']
    Y = data['y']
    Y = np.ravel(Y)

    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()

    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    # reg = regularization
    l = .1
    tags = 10
    result, X = oneVsAll(X, Y, tags, l)
    np.save('./result.npy', result)
    result = np.load('./result.npy')
    print(calcularAciertos(X, Y, result, len(result)))


main()
