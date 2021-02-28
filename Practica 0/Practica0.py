
import numpy as np
import matplotlib.pyplot as plt
import time

def cuadrado(x):
    return x*x
    
def integra_mc_it(fun, a, b, num_puntos=10000):
    """Calcula la integral de fun entre a y b por Monte Carlo con bucles"""
    count = 0
    eje_x = np.linspace(a,b,num_puntos)
    eje_y= fun(eje_x)
    maximo_fun= max(eje_y)
    for i in range(num_puntos):
        x = np.random.uniform(a, b)
        y = np.random.uniform(0, maximo_fun)
        if y < fun(x):
            count += 1            
    integral = count / num_puntos * (b - a)  * maximo_fun
    return integral 

def integra_mc_vect(fun, a, b, num_puntos=10000):
    """Calcula la integral de fun entre a y b por Monte Carlo sin bucles"""
    eje_x = np.linspace(a,b,num_puntos)
    eje_y= fun(eje_x)
    maximo_fun= max(eje_y)
    x = np.random.uniform(a, b,num_puntos)
    y = np.random.uniform(0, maximo_fun,num_puntos) 
    count = sum(y<fun(x))
    integral = count / num_puntos * (b - a)  * maximo_fun
    return integral 
    
def compara_tiempos():
    """Compara tiempos entre integra_mc_vect y integra_mc_it"""
    times_it = []
    times_vect=[]
    num_puntos=10000
    sizes=np.linspace(1,num_puntos,50)
    for size in sizes:
        tic= time.process_time()
        integra_mc_it(cuadrado,1,np.pi)
        toc= time.process_time()
        times_it+=[1000 * (toc-tic)]

        tic_1= time.process_time()
        integra_mc_vect(cuadrado,1,np.pi)
        toc_1= time.process_time()
        times_vect+=[1000 * (toc_1-tic_1)]
    plt.figure()
    plt.scatter(sizes,times_it,c='red',label='it')
    plt.scatter(sizes,times_vect,c='blue',label='vect')
    plt.legend()
    plt.savefig('Practica 0/time.png')


compara_tiempos()