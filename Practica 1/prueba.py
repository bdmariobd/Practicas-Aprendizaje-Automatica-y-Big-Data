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
from mpl_toolkits.mplot3d import Axes3D

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

    x=np.linspace(min(X),max(X),100)
    y= hypothesis(theta_0,theta_1,x)
    plt.plot(x,y)
    plt.show()
    plt.savefig("resultado.pdf")

def cost(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    m = len(X)
    return Aux.sum() / (2 * m)


linear_regression()

data = load_data("./ex1data1.csv")
X = data[:, 0]
Y = data[:, 1]
#Theta_0 [−10, 10] and Theta_1 ∈ [−1, 4]
t0 = np.arange(-10,10,0.1)
t1 = np.arange(-1,4,0.1)
t0,t1 = np.meshgrid(t0,t1)
costs= np.empty_like(t0)

for ix, iy in np.ndindex(t0.shape):
        costs[ix, iy] = cost(X, Y, [t0[ix, iy],t1[ix,iy]])

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(t0,t1,costs,cmap='hot')
plt.contour(t0, t1, costs, np.logspace(-2, 3, 20))
plt.show()
#plt.contour(t0,t1)


