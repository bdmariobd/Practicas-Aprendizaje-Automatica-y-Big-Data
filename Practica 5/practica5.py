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
	return np.dot(X, T)


def cost(T, X, Y, l):
	m = X.shape[0]
	ret = 0
	for row in range(len(X)):
		H = hypothesis(X[row], T)
		ret += np.square(H - Y[row])

	ret = (1 / (2 * m)) * ret
	ret += (l / (2 * m)) * (np.sum(np.square(T[1:])))
	return ret
    
    
def gradient(X, Y, l, T):
	m = X.shape[0]
	D1 = np.zeros(T.shape)

	for row in range(len(X)):
		H = hypothesis(X[row], T)
		D1 += (H - Y[row]) * X[row]
		
	D1 /= m
	# Regularizacion de todos menos j=0
	D1[1:] += (l / m * T[1:])

	return D1


def costAndGrad(X, Y, T, l):
	return cost(T, X, Y, l), gradient(X, Y, l, T)

def get_errors(X,Y, Xval, Yval,l):
    train_errors =  []
    validation_errors = []
    m= X.shape[0]
    for i in range(1,m):
        T = np.zeros(X.shape[1])
        thetas = opt.minimize(fun=cost, x0=T, args=(X[0:i],Y[0:i],l)).x
        train_errors.append(cost(thetas,X[0:i],Y[0:i],l))
        validation_errors.append(cost(thetas,Xval,Yval,l))
        
    return (train_errors ,validation_errors)

def polynomial(X,p):
    polynomial = X
    for i in range(1, p):
        polynomial = np.column_stack((polynomial, np.power(X, i+1)))   
    return polynomial

def normalize(X):
    avg = np.mean(X,axis=0)
    standard_deviation = np.std(X,axis=0)
    normalized = (X - avg) / standard_deviation
    
    return (normalized, avg, standard_deviation)

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


def main():
    data = io.loadmat('./ex5data1.mat')
    X, Y = data['X'], data['y']
    Xval, Yval = data['Xval'], data['yval']
    Xtest, Ytest = data['Xtest'], data['ytest']
    plt.plot(X,Y,'x')
    X_ones = np.hstack([np.ones([np.shape(X)[0], 1]), X])
    l = 1
    T = np.array([ 1, 1 ])
	
    print(costAndGrad(X_ones, Y, T, l))
    res = (opt.minimize(fun=cost, x0=T, args=(X_ones, Y, 0))).x
    
    theta_0, theta_1 = res[0], res[1]
    x = np.linspace(min(X), max(X), 100)
    y = theta_0 + theta_1 * x
    plt.plot(x, y, label=('y = ' + str(theta_1) + 'x + ' + str(theta_0)))
    plt.suptitle('Result')
    plt.xlabel('Change in water level\'s')
    plt.ylabel('Water flowing out of the dam\'s')
	# plt.savefig('result.png')
    plt.show()
    
    train_errors ,value_errors = get_errors(X, Y, Xval, Yval, 0)
    
    x = np.linspace(1, 11, 11)
    plt.plot(x,train_errors, label='Train')
    x = np.linspace(1, 11, 11)
    plt.plot(x,value_errors, label='CrossVal')
    plt.legend()
    plt.suptitle('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.show()
    
    X_normalized, avg, standard_deviation  = normalize(polynomial(X,8))
    X_normalized_ones=np.hstack([np.ones([X_normalized.shape[0],1]),X_normalized])
    
    T = np.zeros(X_normalized_ones.shape[1])

    res = (opt.minimize(fun=cost, x0=T, args=(X_normalized_ones, Y, 0))).x
    

    plt.plot(X,Y,'x')
    plt.plot(polyline, res)
    plt.show()
    print(res)
    


    



main()
