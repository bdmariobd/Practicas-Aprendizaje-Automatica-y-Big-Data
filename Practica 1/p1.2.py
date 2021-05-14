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

def calculate_normal(X,Y):
    transpose = np.transpose(X)
    return np.linalg.inv(transpose.dot(X)).dot((transpose.dot(Y)))

def cost(X, Y, T):
    XTY = np.matmul(X, T) - Y
    return 1 / (2 * np.shape(X)[0]) * (np.matmul(np.transpose(XTY), XTY))

def normalize(X):
    ranges = [ 0 ]
    averages = [ 1 ]
    XNorm = np.copy(X)
    for i in range(1, np.shape(X)[1]):
        col = XNorm[:, i]
        ran = np.max(col) - np.min(col)
        avg = np.average(col)
        col -= avg
        col /= ran
        ranges.append(ran)
        averages.append(avg)
    averages = np.array(averages)
    ranges = np.array(ranges)
    return XNorm, averages, ranges

def main():

    data = getMat('./ex1data2.csv')
    X = data[:, :-1]
    X = np.hstack([np.ones([len(X), 1]), X])
    Y = data[:, -1]
    XNorm,averages,ranges = normalize(X)
    costs = []
    alpha = 0.1
    iterations = 1500
    T = np.zeros(np.shape(X)[1])

    for i in range(iterations):
        T = gradient(XNorm, Y, T, alpha)
        costs.append(cost(XNorm, Y, T))

    plt.suptitle('Cost')
    plt.xlabel('Iterations')
    plt.ylabel('J(theta)')
    plt.plot(np.arange(0, iterations), costs)
    plt.savefig('costs.png')
    plt.show()

    result = calculate_normal(X,Y)
    print('Predicciones del CSV usando grandiente y la ecuacion normal: ')
    for i in range(len(X)):
        print('Gradiente:', np.dot(np.transpose(T), XNorm[i]))
        print('Ec. normal: ', np.dot(np.transpose(result), X[i]))
        print('Valor del ejemplo:', Y[i])
        print('------------------')

    print("Casa con una superficie de 1.650 pies cuadrados y 3 habitaciones:")
    print("Ec.normal: ", result[0] + result[1] * 1650 + result[2] * 3)
    print("Gradiente: ",T[0] + T[1] * ((1650-averages[1])/ranges[1]) + T[2] * ((3-averages[2])/ranges[2]))


main()
