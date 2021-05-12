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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def getMat(file_name):
    return read_csv(file_name, header=None).to_numpy().astype(float)


# Prediction of the value y. In this case using linear regression.
def hypothesis(X, theta_0, theta_1):
    return X * theta_1 + theta_0
    

# Cost function J(theta), measures how good are our guesses
def cost(X, Y, theta_0, theta_1):
    return (1 / (2 * len(X))) * (((hypothesis(X, theta_0, theta_1) - Y)**2).sum())


def gradientDescent(X, Y, iterations, alpha):
    theta_0 = theta_1 = 0
    costs = []

    for i in range(iterations):
        H = hypothesis(X, theta_0, theta_1)
        temp_0 = theta_0 - (alpha / len(X)) * (H - Y).sum()
        temp_1 = theta_1 - (alpha / len(X)) * ((H - Y) * X).sum()
        theta_0 = temp_0
        theta_1 = temp_1
        costs.append(cost(X, Y, theta_0, theta_1))

    plt.suptitle('Cost')
    plt.xlabel('Iterations')
    plt.ylabel('J(theta)')
    plt.plot(np.arange(0, iterations), costs)
    plt.savefig('costs.png')
    plt.show()
    return theta_0, theta_1


def plot_surfaces(theta_0_, theta_1_, theta_0_range, theta_1_range, X, Y):
    step = .1
    theta_0 = np.arange(theta_0_range[0], theta_0_range[1], step)
    theta_1 = np.arange(theta_1_range[0], theta_1_range[1], step)
    theta_0, theta_1 = np.meshgrid(theta_0, theta_1)
    Cost = np.empty_like(theta_0)
    for i_x, i_y in np.ndindex(theta_0.shape):
        Cost[i_x, i_y] = cost(X, Y, theta_0[i_x, i_y], theta_1[i_x, i_y])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(theta_0, theta_1, Cost, cmap=cm.coolwarm)
    plt.show()
    
    plt.contour(theta_0, theta_1, Cost, np.logspace(-2, 3, 20), colors='red')
    plt.scatter(theta_0_, theta_1_)
    plt.show()


def main():
    data = getMat('./ex1data1.csv')
    X = data[:, 0]
    Y = data[:, 1]
    alpha = 0.0001
    iterations = 1500

    theta_0, theta_1 = gradientDescent(X, Y, iterations, alpha)
    x = np.linspace(min(X), max(X), 100)
    y = theta_0 + theta_1 * x
    
    plt.suptitle('Result')
    plt.xlabel('City population in 10k\'s')
    plt.ylabel('Income in 10k\'s')
    plt.plot(X, Y, 'x')
    plt.plot(x, y, label=('y = ' + str(theta_1) + 'x + ' + str(theta_0)))
    plt.legend()
    plt.savefig('result.png')
    plt.show()

    plot_surfaces(theta_0, theta_1, [-10, 10], [-1, 4], X, Y)
    

main()
