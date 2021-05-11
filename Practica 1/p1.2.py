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
from pandas.io.parsers import read_csv

def getMat(file_name):
    return read_csv(file_name, header=None).to_numpy().astype(float)


def gradient(X, Y, T, alpha):
    new_T = T
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, T)
    aux = (H - Y)
    for i in range(n):
        aux_i = aux * X[:, i]
        new_T[i] -= (alpha / m) * aux_i.sum()
    return new_T


def cost(X, Y, T):
    XTY = np.matmul(X, T) - Y
    return 1 / (2 * np.shape(X)[0]) * (np.matmul(np.transpose(XTY), XTY))


def main():

    data = getMat('./ex1data2.csv')
    X = data[:, :-1]
    X = np.hstack([np.ones([len(X), 1]), X])
    Y = data[:, -1]

    ranges = [ 0 ]
    averages = [ 1 ]
    for i in range(1, np.shape(X)[1]):
        col = X[:, i]
        ran = np.max(col) - np.min(col)
        avg =  np.average(col)
        col -= avg
        col /= ran
        ranges.append(ran)
        averages.append(avg)

    # print(X, ranges, averages)

    costs = []
    alpha = 0.01
    iterations = 1000
    T = np.zeros(np.shape(X)[1])
    for i in range(iterations):
        T = gradient(X, Y, T, alpha)
        costs.append(cost(X, Y, T))

    plt.suptitle('Cost')
    plt.xlabel('Iterations')
    plt.ylabel('J(theta)')
    plt.plot(np.arange(0, iterations), costs)
    plt.savefig('costs.png')
    plt.show()

    print(T)
    for i in range(len(X)):
        print(np.dot(np.transpose(T), X[i]))

main()
