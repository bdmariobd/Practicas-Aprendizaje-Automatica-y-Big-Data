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

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
import sklearn.preprocessing


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def hypothesis(X, T):
    return sigmoid(np.matmul(T.T, X))


def nn(X, theta1, theta2):
    a1 = X
    a2 = np.matmul(theta1, a1.T)
    a2 = sigmoid(a2).T
    a2 = np.hstack([np.ones([np.shape(a2)[0], 1]), a2])
    a3 = np.matmul(theta2, a2.T)
    a3 = sigmoid(a3).T

    return a3


def calcularAciertos(X, Y, H):
    aciertos = 0
    for row in range(len(H)):
        guess = np.argmax(H[row]) + 1
        if guess == Y[row]:
            aciertos += 1

    return aciertos / len(H) * 100


def main():
    data = loadmat('./ex3data1.mat')
    X = data['X']
    Y = data['y']
    Y = np.ravel(Y)

    weights = loadmat('./ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    X = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    result = nn(X, theta1, theta2)
    print(calcularAciertos(X, Y, result))


main()
