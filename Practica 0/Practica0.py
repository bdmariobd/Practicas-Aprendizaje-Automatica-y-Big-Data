import numpy as np

def cuadrado(x):
    return x*x

def nada(x):
    return x


def calculo_lento(a, b, M, num_puntos, fun):
    count = 0
    for i in range(num_puntos):
        x = np.random.uniform(a, b)
        y = np.random.uniform(0, M)
        if y < fun(x):
            count += 1
            
    integral = count / num_puntos * (b - a)  * M  
    
def calculo_rapido(a, b, M, num_puntos, fun):
 
    x = np.random.uniform(a, b,num_puntos)
    y = np.random.uniform(0, M,num_puntos)
    
    
            
    integral = count / num_puntos * (b - a)  * M  
    
def integra_mc(fun, a, b, num_puntos=10000):
    print("prueba")
    
    
####juan

import numpy as np

def integra_mc_it(fun, a, b, puntos, max, num_puntos=1000):
    debajo = 0
    for x, y in puntos:
        if(y < cuadrado(x)):
            debajo += 1
    return debajo / num_puntos * (b - a) * max

def integra_mc_vect(fun,  a, b, max, num_puntos=1000):
    pass

def cuadrado(x):
    return x * x

def main():
    a = 0
    b = 50
    num_puntos = 10000
    valores = np.arange(a, b, 1)
    valores = cuadrado(valores)
    max = valores.max()

    puntos = np.column_stack((np.random.uniform(a, b, num_puntos), np.random.uniform(0, max, num_puntos)))

    print(puntos)

    print(valores)

    print(integra_mc_it(cuadrado, a, b, puntos, max, num_puntos))

if name == 'main':
    main()