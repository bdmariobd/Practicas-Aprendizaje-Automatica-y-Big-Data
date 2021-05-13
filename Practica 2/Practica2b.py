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

# Obtiene un vector con los índices de los ejemplos positivos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.preprocessing
from pandas.io.parsers import read_csv
import scipy.optimize as opt


def load_data(file_name):
    return read_csv(file_name, header=None).values.astype(float)


def print_data(X, Y):
    pos = np.where(Y == 1)
    # Dibuja los ejemplos positivos
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    pos = np.where(Y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    plt.savefig("data.pdf")
    plt.legend(
        handles=[
            mpatches.Patch(color='black', label='y = 1'),
            mpatches.Patch(color='blue', label='y = 0')
        ])


def sigmoide(Z):
    return 1 / (1 + np.exp(-Z))


def coste(theta, X, Y, l):
    m = np.shape(X)[0]
    H = sigmoide(np.matmul(X, np.transpose(theta)))
    
    l1 = np.transpose(np.log(H))
    l2 = np.transpose(np.log(1 - H))

    ret = (-1 / m) * ((np.matmul(l1, Y)) + (np.matmul(l2, (1 - Y))))
    return ret + l / (2 * m) * np.sum(H * H)


def gradiente(theta, X, Y, l):
    m = np.shape(X)[0]
    H = sigmoide(np.matmul(X, np.transpose(theta)))
    ret = (1 / m) * np.matmul(np.transpose(X), H - Y)
    return ret + (l / m) * theta


def plot_decisionboundary(X, Y, theta, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    	np.linspace(x2_min, x2_max))
    h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(),
    	xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.savefig("boundary.pdf")


def porcentaje_aciertos(X, Y, theta):
    aciertos = 0
    j = 0
    print(X, theta)
    for i in range(len(X)):
        pred = sigmoide(np.dot(X[i], theta))
        if pred >= 0.5 and Y[j] == 1:
            aciertos += 1
        elif pred < 0.5 and Y[j] == 0:
            aciertos += 1
        j += 1
    return aciertos / len(Y) * 100


if __name__ == '__main__':
    datos = load_data('./ex2data2.csv')
    X = datos[:, :-1]
    Y = datos[:, -1]

    print_data(X, Y)

    mapFeature = sklearn.preprocessing.PolynomialFeatures(6)
    mapFeatureX = mapFeature.fit_transform(X)
    theta = np.zeros(mapFeatureX.shape[1])
    l=1
    
    result = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(mapFeatureX, Y,l))
    print(result)
    theta_opt = result[0]
    data = np.delete(mapFeatureX, 0, axis=1)

    plot_decisionboundary(data, Y, theta_opt,mapFeature)
    plt.show()
    print("Prediccion con un porcentaje de aciertos de:",
        porcentaje_aciertos(mapFeatureX, Y, theta_opt))

