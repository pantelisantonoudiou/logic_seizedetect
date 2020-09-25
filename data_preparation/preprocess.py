# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:03:25 2020

@author: Pante
"""

import numpy as np
from tqdm import tqdm
from scipy.signal import butter, ellip, cheb2ord, cheby2, zpk2sos, sosfiltfilt


###### --------------------- Batch process functions ----------------- ######
def preprocess_data(data,  clean = True, filt = True, verbose = 1):
    """
    preprocess_data(data,  clean = True, filt = True)
     
    data : 3d ndarray, (1d = segments, 2d = time, 3d = channels)
    clean : Bool, optional, The default is True.
    filt : Bool, optional, The default is True.
    verbose: 1 or 0

    Returns
    -------
    data : 3d ndarray, (1d = segments, 2d = time, 3d = channels)
    """
    dim = data.shape # get data dimensions
    
    if verbose == 1:
        if clean == True: # if true remove outliers
            print('Removing Outliers...')
            for i in range(data.shape[2]):
                temp = clean_signal(data[:,:,i].flatten(), threshold = 25)
                data[:,:,i] = temp.reshape((dim[0], dim[1]))
    
        if filt == True:
            print('Filtering...')
            for i in range(data.shape[2]):
                # high-pass filter data
                # UNCOMMENT LINES BELOW FOR HIGH-PASS
                data[:,:,i] = batch_filter(data[:,:,i], butter_highpass, 
                                                      freq_cutoff = [2], verbose = verbose)   
                # band-pass filter data
                # UNCOMMENT LINES BELOW FOR BAND-PASS
                # data[:,:,i] = batch_filter(data[:,:,i], cheby_bandpass, 
                #                                       freq_cutoff = [6,12], verbose = verbose)
    elif verbose == 0:

        if clean == True: # if true remove outliers
            for i in range(data.shape[2]):
                temp = clean_signal(data[:,:,i].flatten(), threshold = 25)
                data[:,:,i] = temp.reshape((dim[0], dim[1]))
    
        if filt == True:
            for i in range(data.shape[2]):
                # high-pass filter data
                data[:,:,i] = batch_filter(data[:,:,i], butter_highpass, 
                                                      freq_cutoff = [2], verbose = verbose)
    return data


def clean_signal(data, threshold = 25):
    """
    clean_signal(data, threshold = 25)
    Removes outliers and replaces with mean

    Parameters
    ----------
    data : 1D signal, numpy array
    threshold : Real number, The default is 25.

    Returns
    -------
    clean_data : 1D signal, numpy array

    """
    
    # copy data
    clean_data = np.copy(data)
    
    # remove mean
    clean_data = clean_data - np.mean(clean_data)
    
    # get index where data exceed threhold based on standard deviation
    idx = np.where(np.abs(data) > (threshold * np.std(data)))
    
    # replace with mean
    clean_data[idx] = np.mean(clean_data)
    
    return clean_data

def batch_filter(data, filt_func, freq_cutoff, fs = 100, verbose = 1):
    """
    batch_filter(data)

    Parameters
    ----------
    data : 2d numpy array
    filt_func: filter object
    freq_cutoff: list with frequency cutoff(s)
    verbose: 1 or 0

    Returns
    -------
    filt_data : 2d numpy array

    """
    
    # Init data matrix
    filt_data = np.zeros(data.shape)
    filt_data = filt_data.astype(np.double)
    
    if verbose == 1:
        for i in tqdm(range(data.shape[0])):# iterate over segments
            filt_data[i,:] = filt_func(data[i,:], freq_cutoff, fs=fs)
            
    elif verbose == 0:
        for i in range(data.shape[0]):# iterate over segments
            filt_data[i,:] = filt_func(data[i,:], freq_cutoff, fs=fs)
        
        
    return filt_data



###### -----------------Individual filters ------------------------- ######

def butter_highpass(data, cutoff, fs, order = 5):
    """
    butter_highpass(data, cutoff, fs, order = 5)

    Parameters
    ----------
    data : 1d ndarray, signal 
    cutoff : Float, cutoff frequency
    fs : Int, sampling rate
    order : Int, filter order. The default is 5.

    Returns
    -------
    y : 1d ndarray, filtered signal 

    """
    nyq = 0.5 * fs               # Nyquist Frequency (Hz)
    normal_cutoff = cutoff[0] / nyq # Low-bound Frequency (Normalised)
    
    # Design filter
    z,p,k = butter(order, normal_cutoff, btype='high', analog=False, output ='zpk') 
    
    sos = zpk2sos(z,p,k)         # Convert to second order sections
    y = sosfiltfilt(sos, data)   # Filter data
    return y

def butter_bandpass(data, cutoff, fs, order = 10):
    """
    butter_bandpass(data, cutoff, fs, order = 10)

    Parameters
    ----------
    data : 1d ndarray, signal 
    cutoff : List [lowcut = Float, low bound frequency limit,  highcut = Float, upper bound frequency limit]
    fs : Int, sampling rate
    order : Int, filter order. The default is 5.

    Returns
    -------
    y : 1d ndarray, filtered signal 

    """
    nyq = 0.5 * fs                      # Nyquist Frequency (Hz)
    low = cutoff[0] / nyq                  # Low-bound Frequency (Normalised)
    high = cutoff[1] / nyq                # Upper-bound Frequency (Normalised)
    
    # Design filter
    z,p,k = butter(order, [low, high], btype='band', analog=False, output ='zpk')
    
    # Convert to second order sections
    sos = zpk2sos(z,p,k) 
    
    # Filter data
    y = sosfiltfilt(sos, data)
    return y


def cheby_bandpass(data, cutoff, fs, Rs = 100):
    """
    cheby_bandpass(data, flow, fhigh, fs, , Rs = 100)

    Parameters
    ----------
    data :  1d ndarray, signal 
    cutoff : List [lowcut = Float, low bound frequency limit,  highcut = Float, upper bound frequency limit]
    fs : Int, sampling rate
    Rs: Int, stopband attenuation in Db, Optional, Default value = 100.

    Returns
    -------
    y : 1d ndarray, filtered signal 

    """
    flow = cutoff[0]; fhigh = cutoff[1]  
    Fn = fs/2                                          # Nyquist Frequency (Hz)
    Wp = [flow/Fn,   fhigh/Fn]                         # Passband Frequency (Normalised)
    Ws = [(flow-1)/Fn,   (fhigh+1)/Fn]                 # Stopband Frequency (Normalised)
    Rp = 1                                             # Passband Ripple (dB)
    Rs = 100     
    
    # Get Filter Order
    n, Ws = cheb2ord(Wp,Ws,Rp,Rs);
    
    # Design Filter
    z,p,k = cheby2(n,Rs,Ws,btype = 'bandpass', analog=False, output='zpk')
    
    # Convert to second order sections
    sos = zpk2sos(z,p,k); 
    
    # filter data
    y = sosfiltfilt(sos,data)
    return y    


def elip_bandpass(data, cutoff, fs, order=10): # ripple on pass band! will not be used
    """
    elip_bandpass(data, cutoff, fs, order=10)
    
    data :  1d ndarray, signal 
    cutoff : List [lowcut = Float, low bound frequency limit,  highcut = Float, upper bound frequency limit]
    fs : Int, sampling rate
    order: Int, filter order, optional. Deafulat value = 10.
    """

    rp = 1; rs = 150
    # need to double check
    z,p,k = ellip(order, rp, rs, cutoff, 'bandpass', analog=False, output='zpk', fs = fs)
    
    # Convert to second order sections
    sos = zpk2sos(z,p,k); 
    
    # filter data
    y = sosfiltfilt(sos, data)
    return y
  ######## -------------------------------------------------------------------------- ########     


### PRINT FILTER RESPONSES ###
        
# ### need to change filter function output to sos for this function to work !!
# def test_run(fs=100, order= 10): 
    
#     from scipy.signal import sosfreqz
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Sample rate and desired cutoff frequencies (in Hz).
#     lowcut = 6
#     highcut = 12

#     # Plot the frequency response for a few different orders.
#     plt.figure(1)
#     plt.clf()

#     sos = butter_highpass([],lowcut, fs, order)
#     w, h = sosfreqz(sos, worN=2000)
#     # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="butter")
#     h = np.abs(h); h[h == 0] = h[1];
#     plt.plot(w, 20 * np.log10(h), label='butter')

#     sos = butter_bandpass([],lowcut, highcut, fs, order)
#     w, h = sosfreqz(sos, worN=2000, fs = fs)
#     # plt.plot(w, abs(h), label="butter")
#     h = np.abs(h); h[h ==0] = h[1];
#     plt.plot(w, 20 * np.log10(h), label='butter')

#     sos = cheby_bandpass([],lowcut, highcut, fs)
#     w, h = sosfreqz(sos, worN=2000, fs = fs)
#     # plt.plot(w, abs(h), label="cheby")
#     h = np.abs(h); h[h ==0] = h[1];
#     plt.plot(w, 20 * np.log10(h), label='cheby2')
    
    
#     # sos = elip_bandpass([],lowcut, highcut, fs, order)
#     # w, h = sosfreqz(sos, worN=2000)
#     # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="Eliptic")
    

#     # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
#     #           '--', label='sqrt(0.5)')
    
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Gain')
#     plt.grid(True)
#     plt.legend(loc='best')
#     plt.title("order = %d" % order)

# if __name__ == '__main__':
#     test_run(fs=100,order = 50)









