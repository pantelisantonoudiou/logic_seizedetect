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


@timing
def f_serial(array):
    s = 0
    for i in range(array.shape[0]):
        s += array[i]*array[i]
    return s

@jit(nopython = True, nogil = False) 
def f_basic(x):
    x*x
    
@timing
def f_parallel(array):    
    s = 0
    p = Pool()
    s = p.map(f_basic, array)
    return s

@timing       
@jit(nopython = True, nogil = False)        
def f_serial_numba(array):
    s = 0
    for i in range(array.shape[0]):
        s += array[i]*array[i]
    return s

@timing
@jit(nopython = True, parallel = True, nogil = False)        
def f_parallel_numba(array):
    s = 0
    for i in prange(array.shape[0]):
        s += array[i]*array[i]
    return s

@timing
@jit(nopython = True, parallel = True, nogil = True)        
def f_parallel_numba_nogil(array):
    s = 0
    for i in prange(array.shape[0]):
        s += array[i]*array[i]
    return s      


if __name__ == '__main__':
    
    
    # print(p.map(f, array))
  
    
    array = np.random.rand(1000000)
    f_serial(array)
    f_parallel(array)
    f_serial_numba(array)
    f_parallel_numba(array)
    f_parallel_numba_nogil(array)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    