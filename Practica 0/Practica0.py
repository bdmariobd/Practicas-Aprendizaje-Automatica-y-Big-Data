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
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
import time


def cuadrado(x):
    return x * x


def integra_mc_it(fun, a, b, num_puntos=10000):
    """Calcula la integral de fun entre a y b por Monte Carlo con bucles"""
    count = 0
    eje_y = fun(np.linspace(a, b, num_puntos))
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
    eje_y = fun(np.linspace(a, b, num_puntos))
    maximo_fun = max(eje_y)

    x = np.random.uniform(a, b, num_puntos)
    y = np.random.uniform(0, maximo_fun, num_puntos)
    count = sum(y < fun(x))

    integral = count / num_puntos * (b - a)  * maximo_fun
    return integral


def compara_tiempos():
    """Compara tiempos entre integra_mc_vect y integra_mc_it"""

    num_puntos = 100000
    a = 1
    b = 3

    times_it = []
    times_vect = []

    sizes = np.linspace(1, num_puntos, 50)
    for i in sizes:
        t1 = time.process_time()
        integra_mc_it(cuadrado, a, b, int(i))
        t2 = time.process_time()
        elapsed_time = 1000 * (t2 - t1)
        times_it += [elapsed_time]

        t1 = time.process_time()
        integra_mc_vect(cuadrado, a, b, int(i))
        t2 = time.process_time()
        elapsed_time = 1000 * (t2 - t1)
        times_vect += [elapsed_time]

    plt.figure()
    plt.scatter(sizes, times_it, c='red', label='it')
    plt.scatter(sizes, times_vect, c='blue', label='vect')
    plt.legend()
    plt.xlabel('num_puntos')
    plt.ylabel('tiempo')
    plt.savefig('./time.png')
    plt.close()

    # print(integra_mc_it(cuadrado, a, b, num_puntos))
    # print(integra_mc_vect(cuadrado, a, b, num_puntos))

compara_tiempos()

