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
import scipy.io as io
import sklearn.preprocessing


def hypothesis(X, T):
	return X * T[0][0] + T[0][1]


def cost(X, Y, l, T):
    H = hypothesis(X, T)
    m = X.shape[0]
    ret = (1 / (2 * m)) * (np.sum(np.square(H - Y)))
    ret += (l / (2 * m)) * (np.sum(np.square(T[:, 1:])))
    return ret
    
    
def gradient(X, Y, l, T):
	m = X.shape[0]
	H = hypothesis(X, T)
	D1 = np.zeros(T.shape)
	A1 = np.hstack([np.ones([m, 1]), X])

	for t in range(m):
		a1t = A1[t, :]
		ht = H[t, :]
		yt = Y[t]

		d3t = ht - yt
		d2t = np.dot(T.T, d3t)

		D1 += np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])

	D1 *= 1 / m
	# Regularizacion de todos menos j=0
	D1[:, 1:] += (l / m * T[:, 1:])

	return D1


def costAndGrad(X, Y, T, l):
	return cost(X, Y, l, T), gradient(X, Y, l, T)

def main():
	data = io.loadmat('./ex5data1.mat')
	X, Y = data['X'], data['y']
	Xval, Yval = data['Xval'], data['yval']
	Xtest, Ytest = data['Xtest'], data['ytest']
    
	l = 1
	T = np.ones((1, 2))
	print(cost(X, Y, l, T))
	print(costAndGrad(X, Y, T, l))

main()
