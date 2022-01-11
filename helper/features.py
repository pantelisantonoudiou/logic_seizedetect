# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:10:02 2020

@author: Pante
"""
import numpy as np
from numba import jit
from scipy.signal import welch, hilbert
from scipy.fftpack import fft

@jit(nopython = True)
def find_nearest(array, value):
    """
    find_nearest(array, value)
    find the index of the nearest value in array

    Parameters
    ----------
    array : ndarray
    value : Real Number

    Returns
    -------
    idx : Int, index of nearest value

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

@jit(nopython=True)
def moving_average(a, n) :
    ret = np.cumsum(a)
    ret = ret.astype(np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

@jit(nopython=True)
def line_length(signal):
    """
    line_length(signal)
    Measures the line length of a signal

    Parameters
    ----------
    signal : 1D numpy array
    -------
    line_length : np.float
    """
    # initialize line length
    line_lengths = np.zeros(1) #signal.shape[1]
    line_lengths = line_lengths.astype(np.double)
    
    for i in range(1, signal.shape[0]): # loop though signal
        line_lengths += np.abs(signal[i] - signal[i-1]) #signal[i, :] - signal[i-1,:]

    # remove zero-length channels
    line_lengths = line_lengths[np.nonzero(line_lengths)] 
    if line_lengths.size == 0: return 0.0

    # take the median and normalize by clip length
    line_length = np.median(line_lengths) / signal.shape[0]
    return line_length


@jit(nopython = True)
def std(signal):
    """
    std(signal)
    Measures standard deviation of a signal

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.std(signal)

@jit(nopython = True)
def var(signal):
    """
    var(signal)
    Measures variance of a signal

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.var(signal)

@jit(nopython = True)
def rms(signal):
    """
    rms(signal)
    Measures root mean square of a signal

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.sqrt(np.mean(np.square(signal)))

@jit(nopython = True)
def rms_envelope(signal, win = 5):
    """
    rms_envelope(signal, win = 5)
    Returns root mean square envelope of a signal
    based on window win

    Parameters
    ----------
    signal : 1D numpy array
    win : moving window
    Output
    -------
    rms_envelope : 1D numpy array
    """
    env = np.zeros(signal.shape[0])
    for i in range(0, signal.shape[0]):
        env[i] = rms(signal[i:i+win])
    return env

@jit(nopython = True)
def max_envelope(signal, win = 5):
    """
    max_envelope(signal, win = 5)
    Returns max envelope of a signal
    based on window win

    Parameters
    ----------
    signal : 1D numpy array
    win : moving window
    Output
    -------
    max_envelope : 1D numpy array
    """
    env = np.zeros(signal.shape[0])
    for i in range(0,signal.shape[0]): 
        env[i] = np.max(signal[i:i+win])
    return env

@jit(nopython = True)
def get_envelope_max_diff(signal,win = 30):
    """
    get_envelope_max_diff(signal, win = 5)
    Measures the sum of the differences between
    the upper and lower envelopes

    Parameters
    ----------
    signal : 1D numpy array
    win : moving window
    Output
    -------
    max_envelope : 1D numpy array
    """
    up_env = max_envelope(signal,win=win)
    
    low_env = -max_envelope(-signal,win=win)
    
    return np.sum(up_env-low_env)
    

@jit(nopython = True)
def mad(signal):
    """
    mad(signal)
    Measures mean absolute deviation of a signal

    Parameters
    ----------
    signal : 1D numpy array
    """
    return np.mean(np.abs(signal - np.mean(signal)))

@jit(nopython = True)
def energy(signal):
    """
    energy(signal)
    Measures the energy of a signal

    Parameters
    ----------
    signal : 1D numpy array
    -------
    energy : np.float
    """
    return np.sum(np.square(signal)) #change with np.mean to calculate mean energy

@jit(nopython = True)
def autocorr(signal):
    """
    autocorr(signal)
    Measures the autocorrelation of a signal

    Parameters
    ----------
    signal : 1D numpy array
    -------
    auto_corr value at 0 : np.float
    """
    return np.correlate(signal,signal)

# get power spectral density    
def psd(signal, fs = 100, freq = [2,40]):
    
    """
    psd(signal, fs, freq)
    Measure the power of the signal over the signal
    over a frequency range as defined by freq
    
    Parameters
    ----------
    signal : 1D numpy array
    fs: Int, sampling rate
    freq: list, low and upper frequency bounds
    
    Returns
    -------
    power_area : np.float
    
    """ 
    
    # Frequency Boundaries
    f_res = signal.shape[0]/fs
    flow  = int(freq[0]*f_res)
    fup = int(freq[1]*f_res) + 1 # upper boundary
     
    # # multiply the fft by hanning window
    # signal = np.multiply(signal,np.hanning(winsize))
       
    # get power spectrum
    xdft = np.square(np.absolute(fft(signal)))
    
    # normalize to signal
    xdft = xdft * (1/(fs*signal.shape[0]))
       
    # multiply *2 to conserve energy in positive frequencies
    psdx = 2*xdft[0:int(xdft.shape[0]/2+1)] 

    return np.sum(psdx[flow:fup])

# get pwelch, slower
def psd_welch(signal, fs = 100, freq = [5,25]):
    
    """
    psd_welch(signal, fs = 100, freq = [5,25])
    Measure the power of the signal over the signal
    over a frequency range as defined by freq
    
    Parameters
    ----------
    signal : 1D numpy array
    fs: Int, sampling rate
    freq: list, low and upper frequency bounds
    
    Returns
    -------
    power_area : np.float
    
    """ 
    
    # calculate pwelch
    f,p = welch(signal, fs=fs, window='hann')
    
    # get frequency boundaries
    flow = find_nearest(f, freq[0])
    fup = find_nearest(f, freq[1])

    return np.sum(p[flow:fup])


@jit(nopython = True)
def cross_corr(signal1, signal2):
    """
    cross_corr(signal1,signal2)
    Measures the cross correlation of two signals with same size

    Parameters
    ----------
    signal1 : 1D numpy array
    signal2 : 1D numpy array
    -------

    """
    return np.correlate(signal1,signal2)


@jit(nopython = True)
def signal_covar(signal1, signal2):
    """
    signal_covar(signal1, signal2)
    Measures the covariance between two signals with same size

    Parameters
    ----------
    signal1 : 1D numpy array
    signal2 : 1D numpy array
    -------

    """
    return np.cov(signal1,signal2)[0][1]

@jit(nopython = True)
def signal_abs_covar(signal1, signal2):
    """
    signal_abs_covar(signal1, signal2)
    Measures the absolute covariance between two signals with same size

    Parameters
    ----------
    signal1 : 1D numpy array
    signal2 : 1D numpy array
    -------
    """
    
    x1 = np.mean(signal1)
    x2 = np.mean(signal2)
    
    abs_covar = np.zeros(signal1.shape[0])
    
    for i in range(signal1.shape[0]):
        abs_covar[i] = np.abs(signal1[i]-x1) * np.abs(signal2[i]-x2)
        
    return np.sum(abs_covar)/signal1.shape[0]
        

def signal_covar_hilbert(signal1, signal2):
    """
    signal_covar_hilbert(signal1,signal2)
    Measures the covariance between hilbert amp of two signals with same size

    Parameters
    ----------
    signal1 : 1D numpy array
    signal2 : 1D numpy array
    -------
    """
    return np.cov(np.abs(hilbert(signal1)), np.abs(hilbert(signal2)))[0][1]
















