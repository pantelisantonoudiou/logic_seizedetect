# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:47:27 2020

@author: Pante
"""

import time
import numpy as np
from multiprocessing import Pool
from numba import jit, prange

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

def f_parallel(x):
    return x*x

def f_serial(array):
    for i in range(array.shape):
        array[i]*array[i]
        
@jit(nopython = True)        
def f_serial_numba(array):
    for i in range(array.shape):
        array[i]*array[i]

@jit(nopython = True, parallel = True)        
def f_parallel_numba(array):
    for i in prange(array.shape):
        array[i]*array[i]         

# def f2(x):
    

if __name__ == '__main__':
    # p = Pool()
    
    # print(p.map(f, array))
    
    array = np.random.rand(10000)
    timing(f_serial)
    
    
    