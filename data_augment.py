# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:08:30 2020

@author: Pante
"""
import numpy as np
from numba import jit
# import pdb

@jit(nopython=True)  
def sliding_window(signal, win, slide_ratio):
    """
    mat = sliding_window(signal, win, slide_ratio)
    mat = sliding_window(szr_segment, 500, 0.1)
    ----------
    
    Parameters
    ----------
    signal : 1D-numpy array of signal
    win : Int, window size in samples
    slide_ratio : Float, ratio of sliding window / window
    
    Returns
    -------
    mat : 2D-numpy array(rows = segments, columns = time points)
    """
    
    sliding_win = int(win*slide_ratio) # get sliding window
    
    # find end limit of signal divided by window size
    end_limit = signal.shape[0] - (signal.shape[0] % win)
    end_limit -= win # subtract window size
    
    # get start index
    start_idx = np.linspace(0,end_limit, int(end_limit/sliding_win)+1)
    start_idx = start_idx.astype(np.int64) # make integer
    
    # init array
    mat = np.zeros((start_idx.shape[0],win),dtype = np.float64)

    for i in range(start_idx.shape[0]): # loop though index
        mat[i,:] = signal[start_idx[i]:start_idx[i] + win]
    return mat


@jit(nopython=True)  
def sliding_window_multi(signal, win, slide_ratio):
    """
    mat = sliding_window_multi(signal, win, slide_ratio)
    mat = sliding_window_multi(szr_segment, 500, 0.1)
    ----------
    
    Parameters
    ----------
    signal : 2D-numpy array of signal (rows = time, columns = channels )
    win : Int, window size in samples
    slide_ratio : Float, ratio of sliding window / window
    
    Returns
    -------
    mat : 3D-numpy array(1D = segments, 2D = time points, 3D = channels)
    """
    
    sliding_win = int(win*slide_ratio) # get sliding window
    
    # find end limit of signal divided by window size
    end_limit = signal.shape[0] - (signal.shape[0] % win)
    end_limit -= win # subtract window size
    
    # get start index
    start_idx = np.linspace(0,end_limit, int(end_limit/sliding_win)+1)
    start_idx = start_idx.astype(np.int64) # make integer
    
    # init array
    mat = np.zeros((start_idx.shape[0],win, signal.shape[1]),dtype = np.float64)

    for i in range(start_idx.shape[0]): # loop though index
        mat[i,:,:] = signal[start_idx[i]:start_idx[i] + win,:]
    return mat




















