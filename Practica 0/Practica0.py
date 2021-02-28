
import numpy as np
import matplotlib.pyplot as plt
import time

save_img_route= 'Practica 0/time.png'
num_puntos = 100000
a=1
b=np.pi

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
    sizes=np.linspace(1,num_puntos,50)
    for size in sizes:
        tic= time.process_time()
        integra_mc_it(cuadrado,a,b,int(size))
        toc= time.process_time()
        elapsed_time = 1000 * (toc-tic)
        times_it+=[elapsed_time]

        tic_1= time.process_time()
        integra_mc_vect(cuadrado,a,b,int(size))
        toc_1= time.process_time()
        elapsed_time = 1000 * (toc_1-tic_1)
        times_vect+=[elapsed_time]

    plt.figure()
    plt.scatter(sizes,times_it,c='red',label='it')
    plt.scatter(sizes,times_vect,c='blue',label='vect')
    plt.legend()
    plt.savefig(save_img_route)
    plt.close()


compara_tiempos()