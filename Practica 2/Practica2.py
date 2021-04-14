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
from pandas.io.parsers import read_csv
import scipy.optimize as opt

def load_data(file_name):
    return read_csv(file_name, header=None).values.astype(float)


def print_data(X, Y):
    pos = np.where(Y == 1)
    # Dibuja los ejemplos positivos
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    pos = np.where(Y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', c='b')
    #plt.show()
    plt.savefig("data.pdf")


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def coste(theta, X, Y):
    m = len(X)
    H = sigmoide(np.matmul(X, theta))
    cost = (- 1 / m) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost


def gradiente(theta, X, Y):
    H = sigmoide(np.matmul(X, theta))
    grad = (1 / len(Y)) * np.matmul(X.T, H - Y)
    return grad

def pinta_frontera_recta(X, Y, theta):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=2, colors='green')

def porcentaje_aciertos(X, Y, theta):
    aciertos = 0
    j = 0
    for i in X:
        prod = sigmoide(np.dot(i, theta))
        if prod >= 0.5 and Y[j] == 1:
            aciertos += 1
        elif prod < 0.5 and Y[j] == 0:
            aciertos += 1
    return aciertos / len(Y) * 100

if __name__ == '__main__':
    datos = load_data('./ex2data1.csv')
    X = datos[:, :-1]

    Y = datos[:, -1]


    print_data(X, Y)
    theta = np.zeros(3)
    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X])
    nX = datos[:, :-1]
    result = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(X, Y))
    theta_opt = result[0]
    pinta_frontera_recta(nX, Y, theta_opt)
    plt.show()
    print("Prediccion con un porcentaje de aciertos de:", porcentaje_aciertos(X, Y, theta_opt))

