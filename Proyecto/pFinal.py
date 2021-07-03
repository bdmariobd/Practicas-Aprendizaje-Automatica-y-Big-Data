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
from pandas.io.parsers import read_csv
import scipy.optimize as opt
import re
import pandas as pd

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

                
def main():
    #Visualizacion de los datos
    
    datos = load_data('./MatchTimelinesFirst15.csv')
    datos = datos.drop(['index','matchId', 'blueDragonKills', 'redDragonKills'], axis=1)
    pd.set_option('display.max_columns',500)
    print(datos.describe(include='all'))
    datos.style
    datos.hist(figsize=(10,10))
    plt.tight_layout()
    plt.show()
    
    
    #Lectura de los datos
    
    data= datos.values
    header = datos.columns
    print(header.values)
    X = data[:,1:]
    Y = data[:, 0:1]
    
    #print_all_grapfs(X,Y,header)
    X_normalized = normalize(X)[0]
    
    
    
    
    
    
    
            
main()