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
import matplotlib.patches as mpatches
from sklearn.utils import shuffle
from pandas.io.parsers import read_csv
import scipy.optimize as opt
import re
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

def load_data(file_name):
    return read_csv(file_name)     
 
        
def data_visualization(X,Y):
    X.hist(figsize=(10,10))
    plt.tight_layout()
    
    
def print_data(X, Y, index, header):
    pos = np.where(Y == 1)
    # Dibuja los ejemplos positivos
    plt.scatter(X[pos, index[0]-1], X[pos, index[1]-1], marker='.', c='blue',s=0.01)
    pos = np.where(Y == 0)
    plt.scatter(X[pos, index[0]-1], X[pos, index[1]-1], marker='.', c='red', s=0.01)
    plt.legend(
        handles=[
            mpatches.Patch(color='blue', label='Blue win'),
            mpatches.Patch(color='red', label='Red win')
        ])
    plt.xlabel(header[index[0]])
    plt.ylabel(header[index[1]])
    plt.show()
    
    
def plot_decisionboundary(X, Y, theta, poly):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    	np.linspace(x2_min, x2_max))
    h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(),
    	xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

    
def print_all_grapfs(X,Y,header):
    for i in range(1,len(header.values)):
        for j in range(1,len(header.values)):
            if(i!=j):
                print_data(X,Y,[i,j],header)
                
def normalize(X):
    avg = np.mean(X, axis=0)
    standard_deviation = np.std(X, axis=0)
    normalized = (X - avg) / standard_deviation

    return (normalized, avg, standard_deviation)

def scale(X):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)

def sigmoide(Z):
    return 1 / (1 + np.exp(-Z))


def coste(theta, X, Y, l):
    m = np.shape(X)[0]
    H = sigmoide(X.dot(theta))
    
    l1 = np.transpose(np.log(H + 1e-06))
    l2 = np.transpose(np.log(1 - H + 1e-06))

    ret = (-1 / m) * ((np.matmul(l1, Y)) + (np.matmul(l2, (1 - Y))))
    return ret + (l / (2 * m)) * np.sum(H * H)


def gradiente(theta, X, Y, l):
    m = np.shape(X)[0]
    H = sigmoide(X.dot(theta))
    ret = (1 / m) * np.matmul(np.transpose(X), H - Y)
    return ret + (l / m) * theta

def porcentaje_aciertos(X, Y, theta):
    aciertos = 0
    j = 0
    for i in range(len(X)):
        pred = sigmoide(np.dot(X[i], theta))
        if pred >= 0.5 and Y[j] == 1:
            aciertos += 1
        elif pred < 0.5 and Y[j] == 0:
            aciertos += 1
        j += 1
    return aciertos / len(Y) * 100

def test_different_values(X,Y,Theta):
    parameters = [0, 0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,100,150]#,200,250,300]
    plt.xlabel("Regularization term")
    plt.ylabel("Success")
    success=[]
    for i in parameters:
        result = opt.fmin_tnc(disp=0, func=coste, x0=Theta, fprime=gradiente, args=(X, Y,i))
        #print(result)
        theta_opt = result[0]
        perc = porcentaje_aciertos(X, Y, theta_opt)
        success.append(perc)
        print("Regularizacion:", i,", prediccion con un porcentaje de aciertos de:", perc)
    plt.scatter(parameters,success)
    plt.plot(parameters,success)
    
def get_errors(X, Y, Xval, Yval, l):
    train_errors = []
    validation_errors = []
    m = X.shape[0]
    for i in range(1, m, 1000):
        T = np.zeros(X.shape[1])
        thetas = opt.minimize(fun=coste, x0=T, args=(X[0:i], Y[0:i], l)).x
        #thetas= opt.fmin_tnc(disp=0, func=coste, x0=T, fprime=gradiente, args=(X[0:i], Y[0:i],l))[0]
        train_errors.append(coste(thetas, X[0:i], Y[0:i], l))
        validation_errors.append(coste(thetas, Xval, Yval, l))

    return (train_errors, validation_errors)

    
def learning_curve(X,Y,Xval,Yval,l):
    train_errors, value_errors = get_errors(X, Y, Xval, Yval, l)
    x = np.linspace(1, len(train_errors), len(train_errors))
    plt.plot(x, train_errors, label='Train')
    x = np.linspace(1, len(train_errors), len(train_errors))
    plt.plot(x, value_errors, label='CrossVal')
    plt.legend()
    plt.suptitle('Learning curve for linear regression (lambda =' +str(l) + ')')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.show()
    
    
def opt_regresion_parameter(X,Y,Xval,Yval):
    lambdas =[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10,20,100,200]
    train_errors = []
    validation_errors = []

    for i in lambdas:
        initial_thetas = np.zeros(X.shape[1])
        T = opt.minimize(fun=coste, x0=initial_thetas, args=(X,Y,i)).x
        train_errors.append(coste(T,X,Y,i))
        validation_errors.append(coste(T,Xval,Yval,i))
    plt.plot(lambdas,train_errors, label='Train')
    plt.plot(lambdas,validation_errors, label='CrossVal')
    plt.suptitle('Selecting lambda using a cross validation set')
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.legend()
    
    
#SVM
def selectCandSigma(X,Y,Xval,Yval):
    parameters = [0.00001,0.0001,0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    bestC = 0
    bestSigma = 0
    bestScore = 0
    for C in parameters:
        svm = SVC(kernel='linear', C=C)
        svm.fit(X, Y.ravel())
        score = accuracy_score(Yval, svm.predict(Xval))
        print("C=",C, " acierto de ", score)
        if(bestScore < score):
            bestScore = score
            bestC = C
                
    return bestC, bestSigma, bestScore
    
def main():
    #Visualizacion de los datos
    
    datos = load_data('./MatchTimelinesFirst15.csv')
    datos = datos.drop(['index','matchId', 'blueDragonKills', 'redDragonKills'], axis=1)
    datos= datos.sample(frac=1)
    pd.set_option('display.max_columns',500)
    print(datos.describe(include='all'))
    datos.style
    datos.hist(figsize=(10,10))
    plt.tight_layout()
    plt.show()
    #sns.pairplot(datos, corner=True, hue = 'blue_win')
    
    #Lectura de los datos
    
    data= datos.values.astype(float)
    header = datos.columns
    print(header.values)
    X = data[:,1:]
    Y = data[:, 0]
        
    #print_all_grapfs(X,Y,header)
    X_normalized = scale(X)
    X_normalized = np.hstack([np.ones([np.shape(X_normalized)[0], 1]), X_normalized])
    
    
    Xtrain, Ytrain = X_normalized[int(len(X_normalized)*0.60):], Y[int(len(Y)*0.60):] #train theta : 60%
    Xtemp, Ytemp = X_normalized[:int(len(X_normalized)*0.40)], Y[:int(len(Y)*0.40)]
    
    Xval, Yval = Xtemp[int(len(Xtemp)*0.50):], Ytemp[int(len(Ytemp)*0.50):] #validation: 20%
    Xtest, Ytest = Xtemp[:int(len(Xtemp)*0.50)], Ytemp[:int(len(Ytemp)*0.50)] #test: 20%
    
    theta = np.zeros(Xtrain.shape[1])
    l= 1
    #print('Coste: ', str(coste(theta, Xtrain, Ytrain, l)))
    #print('Gradiente: ', np.array2string(gradiente(theta, Xtrain, Ytrain, l)))
    result = opt.fmin_tnc(disp=0, func=coste, x0=theta, fprime=gradiente, args=(Xtrain, Ytrain,l))
    #print(result)
    theta_opt = result[0]
    print("Prediccion con un porcentaje de aciertos de:",
        porcentaje_aciertos(Xtrain,Ytrain, theta_opt))
    
    #test_different_values(Xtrain, Ytrain, theta)
    # learning_curve(Xtrain,Ytrain,Xval,Yval,0)
    # learning_curve(Xtrain,Ytrain,Xval,Yval,1)
    # learning_curve(Xtrain,Ytrain,Xval,Yval,20)
    # learning_curve(Xtrain,Ytrain,Xval,Yval,100)
    # learning_curve(Xtrain,Ytrain,Xval,Yval,200)
    #opt_regresion_parameter(Xtest,Ytest,Xval,Yval)
    
    #SVM
    # svm = SVC(kernel='linear', C=1)
    # svm.fit(Xtrain, Ytrain.ravel())
    # score = accuracy_score(Yval, svm.predict(Xval))
    print()
    C, sigma, score  = selectCandSigma(Xtrain, Ytrain, Xval, Yval)
    print ("Precision con entrenamiento (75% de los casos) y validacion(25% de los casos): ", score)
    print('C=' + str(C) + ' BestSigma =' + str(sigma))
    
    
    
            
main()