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
# SOFTWARE

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
import sklearn.preprocessing

import displayData, checkNNGradients


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def cost(H, X, Y, l, T_1, T_2):
    m = len(Y)
    l1 = np.transpose(np.log(H))
    l2 = np.transpose(np.log(1 - H + 1e-6))
    ret = (1 / m) * (-(l1 * Y.T) - ((1 - Y.T) * l2))
    ret = np.sum(ret)
    ret += (l / (2 * m)) * (np.sum(T_1**2) + np.sum(T_2**2))
    return ret
    

def forward_propagation(X, T1, T2):
    m = X.shape[0]
    
    A1 = np.hstack([np.ones([m, 1]), X])
    Z2 = np.dot(A1, T1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoid(Z2)])
    Z3 = np.dot(A2, T2.T)
    H = sigmoid(Z3)

    return A1, A2, H


# Devuelve una tupla con coste y gradiente
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):
    m = X.shape[0]
    theta_1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],
    	(num_ocultas, (num_entradas + 1)))
    theta_2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):],
    	(num_etiquetas, (num_ocultas + 1)))

    A1, A2, H = forward_propagation(X, theta_1, theta_2)
    D1, D2 = np.zeros(theta_1.shape), np.zeros(theta_2.shape)
    
    # hasta aqui bien?
    
    print(Y, Y.shape)
    
    for t in range(m):
    	a1t = A1[t, :]
    	a2t = A2[t, :]
    	ht = H[t, :]
    	yt = Y[t]
    	
    	d3t = ht - yt
    	d2t = np.dot(theta_2.T, d3t) * (a2t * (1 - a2t))
    	
    	D1 = D1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
    	D2 = D2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])


def main():
    data = loadmat('ex4data1.mat')
    Y = data['y'].ravel()
    X = data['X']
    # X = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    m = len(Y)
    input_size = X.shape[1]
    num_labels = 10

    # Diagonal a unos para poder entrenar
    Y = (Y - 1)
    Y_oneHot = np.zeros((m, num_labels))
    for i in range(m):
        Y_oneHot[i][Y[i]] = 1

    weights = loadmat('ex4weights.mat')
    theta_1, theta_2 = weights['Theta1'], weights['Theta2']

    A1, A2, H = forward_propagation(X, theta_1, theta_2)
    l = 1
    print(cost(H, X, Y_oneHot, l, theta_1, theta_2))

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    reg = 100 # TODO
    
    print(backprop(np.append(np.ravel(theta_1),(np.ravel(theta_2))),
    	input_layer_size, hidden_layer_size, num_labels, X, Y_oneHot, reg))


main()
