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
import sklearn
from sklearn import preprocessing
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
import time
import checkNNGradients
from checkNNGradients import *
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
    scaler = preprocessing.MinMaxScaler().fit(X)
    #scaler = preprocessing.StandardScaler().fit(X) 
    return scaler.transform(X)

def polynomialGradeComparation(X, Y, l):
    times= []
    scores = []
    
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = chopped_dataset(X, Y)
    
    mapFeature = sklearn.preprocessing.PolynomialFeatures(2)
    X_normalized_g2 = mapFeature.fit_transform(X)
    Xtraing2, Ytrain, Xvalg2, Yval, Xtestg2, Ytest = chopped_dataset(X_normalized_g2, Y)
    
    mapFeature = sklearn.preprocessing.PolynomialFeatures(3)
    X_normalized_g3 = mapFeature.fit_transform(X)
    Xtraing3, Ytrain, Xvalg3, Yval, Xtestg3, Ytest = chopped_dataset(X_normalized_g3, Y)
    
    start = time.perf_counter()
    result = opt.fmin_tnc(disp=0, func=coste, x0=np.zeros(Xtrain.shape[1]), fprime=gradiente, args=(Xtrain, Ytrain,l))
    theta_opt = result[0]
    score = porcentaje_aciertos(Xval,Yval, theta_opt)
    print("Prediccion con un porcentaje de aciertos de:", score)
    end = time.perf_counter()
    elapsed_time = end - start
    times.append(elapsed_time)
    scores.append(score)
    
    start = time.perf_counter()
    result = opt.fmin_tnc(disp=0, func=coste, x0=np.zeros(Xtraing2.shape[1]), fprime=gradiente, args=(Xtraing2, Ytrain,l))
    theta_opt = result[0]
    score = porcentaje_aciertos(Xvalg2,Yval, theta_opt)
    print("Prediccion con un porcentaje de aciertos de:", score)
    end = time.perf_counter()
    elapsed_time = end - start
    times.append(elapsed_time)
    scores.append(score)
    
    start = time.perf_counter()
    result = opt.fmin_tnc(disp=0, func=coste, x0=np.zeros(Xtraing3.shape[1]), fprime=gradiente, args=(Xtraing3, Ytrain,l))
    theta_opt = result[0]
    score = porcentaje_aciertos(Xvalg3,Yval, theta_opt)
    print("Prediccion con un porcentaje de aciertos de:", score)
    end = time.perf_counter()
    elapsed_time = end - start
    times.append(elapsed_time)
    scores.append(score)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(['G1', 'G2', 'G3'],times)
    plt.ylabel('Elapsed time')
    plt.show()
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(['G1', 'G2', 'G3'],scores)
    plt.ylabel('Linear regression score')
    plt.show()
    
    
    
    
def chopped_dataset(X,Y):
    Xtrain, Ytrain = X[int(len(X)*0.60):], Y[int(len(Y)*0.60):] #train theta : 60%
    Xtemp, Ytemp = X[:int(len(X)*0.40)], Y[:int(len(Y)*0.40)]
    
    Xval, Yval = Xtemp[int(len(Xtemp)*0.50):], Ytemp[int(len(Ytemp)*0.50):] #validation: 20%
    Xtest, Ytest = Xtemp[:int(len(Xtemp)*0.50)], Ytemp[:int(len(Ytemp)*0.50)] #test: 20%
    
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest
    
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

def test_different_values(X,Y, Xval, Yval):
    parameters = [0, 0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,100,150]#,200,250,300]
    plt.xlabel("Regularization term")
    plt.ylabel("Success")
    success=[]
    for i in parameters:
        result = opt.fmin_tnc(disp=0, func=coste, x0=np.zeros(X.shape[1]), fprime=gradiente, args=(X, Y,i))
        #print(result)
        theta_opt = result[0]
        perc = porcentaje_aciertos(Xval, Yval, theta_opt)
        success.append(perc)
        print("Regularizacion:", i,", prediccion con un porcentaje de aciertos de:", perc)
    plt.scatter(parameters,success)
    plt.plot(parameters,success)
    plt.show()
    
def get_errors(X, Y, Xval, Yval, l):
    train_errors = []
    validation_errors = []
    m = X.shape[0]
    for i in range(1, 100):
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
    lambdas =[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]#,20,100,200]
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
    plt.show()
    
#Neuronal network

def random_weights(Lin, Lout):
    epsilon = (6**1/2) / (Lin+Lout)**1/2
    return np.random.uniform(-epsilon, epsilon, (Lin,Lout))

def forward_propagation(X, T1, T2):
    m = X.shape[0]

    A1 = np.hstack([np.ones([m, 1]), X])
    Z2 = np.dot(A1, T1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoide(Z2)])
    Z3 = np.dot(A2, T2.T)
    H = sigmoide(Z3)

    return A1, A2, H

def cost(X, Y, l, T_1, T_2):
    A1, A2, H = forward_propagation(X, T_1, T_2)
    m = X.shape[0]
    l1 = np.transpose(np.log(H+ 1e-06))
    l2 = np.transpose(np.log(1 - H + 1e-06))
    ret = ((l1.T * -Y) - ((1 - Y) * l2.T))
    ret = np.sum(ret) / m
    ret += (l / (2 * m)) * (np.sum(np.square(T_1[:, 1:])) + np.sum(np.square(T_2[:, 1:])))
    return ret

def gradient(X, Y, l, theta_1, theta_2):
    m = X.shape[0]
    A1, A2, H = forward_propagation(X, theta_1, theta_2)
    D1, D2 = np.zeros(theta_1.shape), np.zeros(theta_2.shape)

    for t in range(m):
        a1t = A1[t, :]
        a2t = A2[t, :]
        ht = H[t, :]
        yt = Y[t]

        d3t = ht - yt
        d2t = np.dot(theta_2.T, d3t) * (a2t * (1 - a2t))

        D1 = D1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        D2 = D2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    D1 *= 1 / m
    D2 *= 1 / m
    # Regularizacion de todos menos j=0
    D1[:, 1:] += (l / m * theta_1[:, 1:])
    D2[:, 1:] += (l / m * theta_2[:, 1:])
    grad = np.concatenate((np.ravel(D1), np.ravel(D2)))

    return grad


def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):
    theta_1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)],
                         (num_ocultas, (num_entradas + 1)))
    theta_2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):],
                         (num_etiquetas, (num_ocultas + 1)))

    return (cost(X, Y, reg, theta_1, theta_2), gradient(X, Y, reg, theta_1, theta_2))

def iterationsScore(params_rn, input_layer_size, hidden_layer_size, num_labels, Xtrain, Ytrain,Xval,Yval,l):
    scores = []
    iterations = np.linspace(10,100,10)
    for i in iterations:
        result = opt.minimize(fun = backprop, x0= params_rn, args=(input_layer_size, hidden_layer_size, num_labels, Xtrain, Ytrain, l),method = 'TNC', options={'maxiter': int(i)} , jac=True)
        
        theta_1= np.reshape(result.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)))
        theta_2 = np.reshape(result.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1)))
        
        """np.save("t1.npy",theta_1)
        np.save("t2.npy",theta_2)
        theta_1 = np.load("t1.npy")
        theta_2 = np.load("t2.npy")"""
        score = calcularAciertos(Xval,Yval,theta_1,theta_2)
        scores.append(score)
        print("El porcentaje de acierto del modelo con redes neuronales es: ", score)
    plt.xlabel('Max iterations')
    plt.ylabel('Score')
    plt.plot(iterations, scores)
    plt.show()
    
def lambdaScore(params_rn, input_layer_size, hidden_layer_size, num_labels, Xtrain, Ytrain,Xval,Yval, iterations):
    scores = []
    l = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    for i in l:
        result = opt.minimize(fun = backprop, x0= params_rn, args=(input_layer_size, hidden_layer_size, num_labels, Xtrain, Ytrain, i),method = 'TNC', options={'maxiter': iterations} , jac=True)
        
        theta_1= np.reshape(result.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)))
        theta_2 = np.reshape(result.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1)))
        
        """np.save("t1.npy",theta_1)
        np.save("t2.npy",theta_2)
        theta_1 = np.load("t1.npy")
        theta_2 = np.load("t2.npy")"""
        score = calcularAciertos(Xval,Yval,theta_1,theta_2)
        scores.append(score)
        print("El porcentaje de acierto del modelo con redes neuronales es: ", score)
    plt.xlabel('Lambda')
    plt.ylabel('Score')
    plt.plot(l, scores)
    plt.show()
    
def calcularAciertos(X, Y, T1, T2):
    aciertos = 0
    j = 0
    tags = len(T2)
    pred = forward_propagation(X, T1, T2)[2]
    for i in range(len(X)):
        maxi = np.argmax(pred[i])
        if Y[i] == maxi:
            aciertos += 1
        j += 1
    return aciertos / len(Y) * 100


#SVM
def selectCandSigmaLinearK(X,Y,Xval,Yval):
    parameters = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]#, 3, 10, 30]
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

def selectCandSigmaGaussK(X,Y,Xval,Yval):
    parameters = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]#, 3, 10, 30]
    bestC = 0
    bestSigma = 0
    bestScore = 0
    for C in parameters:
        for sigma in parameters:
            svm = SVC(kernel='rbf', C=C, gamma=1 / ( 2 * sigma **2))
            svm.fit(X, Y.ravel())
            score = accuracy_score(Yval, svm.predict(Xval))
            print("Sigma=" ,sigma," C=",C, " acierto de ", score)
            if(bestScore < score):
                bestScore = score
                bestC = C
                
    return bestC, bestSigma, bestScore
    
def linearKVSgaussianK(X,Y,Xval,Yval,C):
    plt.legend()
    plt.suptitle('Linear Kernel vs Gaussian Kernel (C =' +str(C) + ')')
    times= []
    start = time.perf_counter()
    svm = SVC(kernel='linear', C=C)
    svm.fit(X, Y.ravel())
    end = time.perf_counter()
    elapsed_time = end - start
    
    score = accuracy_score(Yval, svm.predict(Xval))
    print(elapsed_time, score)
    times.append(elapsed_time)
    
    start = time.perf_counter()
    svm = SVC(kernel='rbf', C=C, gamma=1 / ( 2 * C **2))
    svm.fit(X, Y.ravel())
    score = accuracy_score(Yval, svm.predict(Xval))
    end = time.perf_counter()
    elapsed_time = end - start
    
    print(elapsed_time, score)
    times.append(elapsed_time)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(['Linear', 'Gaussian'],times)
    plt.show()
    
    
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
    
    
    #Lectura de los datos
    
    data= datos.values.astype(float)
    header = datos.columns
    print(header.values)
    X = data[:,1:]
    Y = data[:, 0]
    
    #sns.pairplot(datos, corner=True, hue = 'blue_win')    
    #print_all_grapfs(X,Y,header)
    X_normalized = scale(X)
    
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = chopped_dataset(X_normalized, Y)
    
    
    polynomialGradeComparation(X_normalized, Y, 1)
    l= 1
    
    mapFeature = sklearn.preprocessing.PolynomialFeatures(2)
    X_normalized_g2 = mapFeature.fit_transform(X_normalized)
    Xtraing2, Ytrain, Xvalg2, Yval, Xtestg2, Ytest = chopped_dataset(X_normalized_g2, Y)
    
    mapFeature = sklearn.preprocessing.PolynomialFeatures(3)
    X_normalized_g3 = mapFeature.fit_transform(X_normalized)
    Xtraing3, Ytrain, Xvalg3, Yval, Xtestg3, Ytest = chopped_dataset(X_normalized_g3, Y)
    
    
    """learning_curve(Xtrain,Ytrain,Xval,Yval,0)
    learning_curve(Xtrain,Ytrain,Xval,Yval,0.01)
    learning_curve(Xtrain,Ytrain,Xval,Yval,0.1)
    learning_curve(Xtrain,Ytrain,Xval,Yval,1)
    
    #learning_curve(Xtrain,Ytrain,Xval,Yval,20)
    opt_regresion_parameter(Xtrain,Ytrain,Xval,Yval)"""
    
    """learning_curve(Xtraing2,Ytrain,Xvalg2,Yval,0)
    learning_curve(Xtraing2,Ytrain,Xvalg2,Yval,0.01)
    learning_curve(Xtraing2,Ytrain,Xvalg2,Yval,0.1)
    learning_curve(Xtraing2,Ytrain,Xvalg2,Yval,1)
    learning_curve(Xtraing2,Ytrain,Xvalg2,Yval,20)
    opt_regresion_parameter(Xtraing2,Ytrain,Xvalg2,Yval)"""
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Neuronal network
    
    # checkNNGradients(backprop,0)
    # checkNNGradients(backprop,1)
    
    
    l=0
    #Xtrain = np.delete(Xtrain, 0, axis=1)    
    input_layer_size = Xtrain.shape[1]
    hidden_layer_size = 25
    num_labels = 2
    
          
    theta_1, theta_2 = random_weights(hidden_layer_size,input_layer_size +1), random_weights(num_labels,hidden_layer_size + 1)
    params_rn = np.append(np.ravel(theta_1),(np.ravel(theta_2)))
    
    Y_oneHot = np.zeros((len(Ytrain), num_labels))
    for i in range(len(Ytrain)):
        if Ytrain[i]:
            Y_oneHot[i][1]=1
        else:
            Y_oneHot[i][0]=1
        
    iterationsScore(params_rn, input_layer_size, hidden_layer_size, num_labels, Xtrain, Y_oneHot,Xval,Yval, l)
    lambdaScore(params_rn, input_layer_size, hidden_layer_size, num_labels, Xtrain, Y_oneHot,Xval,Yval, 50)
    
    input_layer_size = Xtraing2.shape[1]
    theta_1, theta_2 = random_weights(hidden_layer_size,input_layer_size +1), random_weights(num_labels,hidden_layer_size + 1)
    params_rn = np.append(np.ravel(theta_1),(np.ravel(theta_2)))
    iterationsScore(params_rn, input_layer_size, hidden_layer_size, num_labels, Xtraing2, Y_oneHot,Xvalg2,Yval, l)
    lambdaScore(params_rn, input_layer_size, hidden_layer_size, num_labels, Xtraing2, Y_oneHot,Xvalg2,Yval, 50)
    
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #SVM
    
    Xtrain = np.hstack([np.ones([np.shape(Xtrain)[0], 1]), Xtrain])
    # svm = SVC(kernel='linear', C=1)
    # svm.fit(Xtrain, Ytrain.ravel())
    # score = accuracy_score(Yval, svm.predict(Xval))
    """C, sigma, score  = selectCandSigmaLinearK(Xtrain, Ytrain, Xval, Yval)
    print ("Lineal kernel: Precision con entrenamiento (60% de los casos) y validacion(20% de los casos): ", score)
    print('C=' + str(C)) #+ ' BestSigma =' + str(sigma))
    
    C, sigma, score  = selectCandSigmaGaussK(Xtrain, Ytrain, Xval, Yval)
    print ("Gauss kernel: Precision con entrenamiento (60% de los casos) y validacion(20% de los casos): ", score)
    print('C=' + str(C) + ' BestSigma =' + str(sigma))
    
    linearKVSgaussianK(Xtrain, Ytrain, Xval, Yval,C)"""
    
    
    
    
            
main()