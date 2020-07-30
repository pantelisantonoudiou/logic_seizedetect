# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:47:27 2020

@author: Pante
"""

import time
import numpy as np
from multiprocessing import Pool
from numba import jit, prange
import matplotlib.pyplot as plt

# def f_basic(x):
#     return x*x
    
# @timing
# def f_parallel(array):    
#     s = 0
#     p = Pool()
#     s = p.map(f_basic, array)
#     return s


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


# @timing
def f_serial_cpython(array):
    s = 0
    for i in range(array.shape[0]):
        s += array[i]*array[i]
    return s

# @timing       
@jit(nopython = True, nogil = False, cache = True)        
def f_serial_numba(array):
    s = 0
    for i in range(array.shape[0]):
        s += array[i]*array[i]
    return s

# @timing
@jit(nopython = True, parallel = True, nogil = False, cache = True)        
def f_parallel_numba(array):
    s = 0
    for i in prange(array.shape[0]):
        s += array[i]*array[i]
    return s

# @timing
@jit(nopython = True, parallel = True, nogil = True, cache = True)        
def f_parallel_numba_nogil(array):
    s = 0
    for i in prange(array.shape[0]):
        s += array[i]*array[i]
    return s


if __name__ == '__main__':
    
    # vector size list
    vector_size = np.array([10**x for x in range(10)])
    
    # function list
    func_array = [f_serial_cpython, f_serial_numba, f_parallel_numba, f_parallel_numba_nogil]
    
    # get function names
    function_labels = [f.__name__ for f in func_array]
    
    # create empty array
    exec_time = np.zeros((len(vector_size), len(func_array)))
    
    # number of repetitions
    reps = 10;  
                            
    for i in range(len(vector_size)): # iterate over increasing array size
        array = np.random.rand(vector_size[i])
        
        for ii in range(len(func_array)): # iterate over different function impementations
        
            toc = 0 # zero the timer
            for iii in range(reps):
                tic = time.time()
                s = func_array[ii](array)
                toc += time.time() - tic
            exec_time[i,ii] = toc
                

plt.plot(np.log(vector_size[3:]), exec_time[3:])
plt.legend(function_labels, loc = 'upper left')
plt.ylabel('Time (ms)')
plt.xlabel('Log input Size')


