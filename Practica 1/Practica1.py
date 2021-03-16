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

alpha = 0.01
it=1500
def load_data(file_name):
    return read_csv(file_name, header=None).to_numpy()

def hypothesis(theta_0, theta_1 , x):
    return theta_0 + theta_1 * x

def linear_regression():
    data = load_data("./ex1data1.csv")
    X = data[:, 0]
    Y = data[:, 1]

    plt.plot(X, Y, "x")
    m = len(X)
    theta_0 = theta_1 = 0
    for i in range(it):
        sum_0 = sum_1 = 0
        for j in range(m):
            sum_0 += hypothesis(theta_0,theta_1,X[j]) - Y[j]
            sum_1 += (hypothesis(theta_0,theta_1,X[j]) - Y[j]) * X[j]
        theta_0 = theta_0 - (alpha/m) * sum_0
        theta_1 = theta_1 - (alpha/m) * sum_1

    x=np.linspace(min(X),max(Y),100)
    y= theta_0 + theta_1 * x
    plt.plot(x,y)
    plt.show()
    plt.savefig("resultado.pdf")

linear_regression()
