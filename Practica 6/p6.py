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
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import process_email, get_vocab_dict
from process_email import *
from get_vocab_dict import *


def visualize_boundary(X, y, svm, file_name=''):
     x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
     x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
     x1, x2 = np.meshgrid(x1, x2)
     yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
     pos = (y == 1).ravel()
     neg = (y == 0).ravel()
     plt.figure()
     plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
     plt.scatter(
     X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
     plt.contour(x1, x2, yp)
     plt.show()
     plt.close()
     
     
def selectCandSigma(X,Y,Xval,Yval):
    parameters = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    bestC = 0
    bestSigma = 0
    bestScore = 0
    for C in parameters:
        for sigma in parameters:
            svm = SVC(kernel='rbf', C=C, gamma=1 / ( 2 * sigma **2))
            svm.fit(X, Y)
            score = accuracy_score(Yval, svm.predict(Xval))
            if(bestScore < score):
                bestSigma = sigma
                bestScore = score
                bestC = C
                
    svm = SVC(kernel='rbf', C=bestC, gamma=1 / ( 2 * bestSigma **2))
    svm.fit(X, Y)
    print('C=' + str(bestC) + ' BestSigma =' + str(bestSigma))
    visualize_boundary(X,Y,svm)
    
            
            
def main():
    data1, data2, data3 = io.loadmat('./p6/ex6data1.mat'), io.loadmat('./p6/ex6data2.mat'), io.loadmat('./p6/ex6data3.mat')
    X1, Y1 = data1['X'], data1['y']
    X2, Y2 = data2['X'], data2['y']
    X3, Y3 = data3['X'], data3['y']
    X3val, Y3val = data3['Xval'], data3['yval']
    
    #1.1. Kernel lineal
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X1, Y1)
    visualize_boundary(X1,Y1,svm)
    
    svm = SVC(kernel='linear', C=100.0)
    svm.fit(X1, Y1)
    visualize_boundary(X1,Y1,svm)
    
    #1.2. Kernel gaussiano
    sigma = 0.1
    svm = SVC(kernel='rbf', C=1, gamma=1 / ( 2 * sigma **2))
    svm.fit(X2, Y2)
    visualize_boundary(X2,Y2,svm)
    
    #1.3. Elección de los parámetros C y sigma
    selectCandSigma(X3,Y3,X3val,Y3val)
    
    #2. Detección de spam
    
            
main()