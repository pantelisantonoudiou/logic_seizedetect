# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:29:12 2020

@author: Pante
"""

import os, features, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from build_feature_data import get_data,preprocess_data, get_features_allch
from array_helper import find_szr_idx, match_szrs, merge_close
import matplotlib.pyplot as plt

main_path =  r'W:\Maguire Lab\Trina\2019\07-July\3514_3553_3639_3640'  # 3514_3553_3639_3640  3642_3641_3560_3514
num_channels = [0,1]

# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,)
cross_ch_param_list = (features.cross_corr,)


# create running standard deviation
from numba import jit

@jit(nopython = True)
def running_std_detection(signal, thresh_multiplier, window):
    """
    running_std_detection(signal, thresh_multiplier, window)

    Parameters
    ----------
    signal : 1D ndarray
    thresh_multiplier : Float, Threshold multiplier
    window : Int, Running window size in samples

    Returns
    -------
    bool_idx : 1D ndarray, same shape as signal, boolean index

    """
    
    # create boolean index array for storage
    bool_idx = np.zeros(signal.shape)
    
    # iterate over samples with running threshold
    nmax = signal.shape[0]-window
    for i in range(nmax):  
         threshold = np.mean(signal[i:i+window]) + thresh_multiplier*np.std(signal[i:i+window])
         bool_idx[i] = signal[i]>threshold
         
    # leave threshold constant for remaining samples smaller than window size
    for i in range(nmax, signal.shape[0]):
         bool_idx[i] = signal[i]>threshold
         
    return bool_idx


def file_loop(main_path):
    
    # get data list 
    ver_path = os.path.join(main_path, 'verified_predictions_pantelis')
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
    filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
    
    # create feature labels
    feature_labels = [x.__name__ for x in param_list]; 
    # feature_labels += [x.__name__ +'_2' for x in param_list]
    feature_labels += [x.__name__  for x in cross_ch_param_list]
    feature_labels = np.array(feature_labels)
    
    # create dataframe
    columns = ['true_total', 'total_detected', 'total_exta']
    df = pd.DataFrame(data= np.zeros((len(feature_labels),len(columns))), columns = columns, dtype=np.int64)
    df['Features'] = feature_labels
    
    # create seizure label
    szrs = np.zeros((len(filelist),3,feature_labels.shape[0]))
        
    for i in range(0, len(filelist)): #loop through experiments 

        # get data and true labels
        data, y_true = get_data(main_path,filelist[i],ch_num = num_channels)
        print('->',filelist[i], 'loaded.')
        
        # Clean and filter data
        data = preprocess_data(data,  clean = True, filt = False)
        print('-> data pre-processed.')
        
        # Get features and labels
        x_data, feature_labels = get_features_allch(data,param_list,cross_ch_param_list)
        # get refine data by channel and remap labels
        new_data = np.multiply(x_data[:,0:len(param_list)],x_data[:,len(param_list):x_data.shape[1]-len(cross_ch_param_list)])
        x_data = np.concatenate((new_data, x_data[:,x_data.shape[1]-1:]), axis=1)
        print('-> features extracted')
        
        # Normalize data
        x_data = StandardScaler().fit_transform(x_data)

        for ii in range(x_data.shape[1]): # iterate through parameteres
        
            # get boolean index
            # y_pred = running_std_detection(x_data[:,ii] , 5, int(60/5)*120)
            y_pred = x_data[:,ii]> (np.mean(x_data[:,ii]) + 5*np.std(x_data[:,ii]))
            
            # get number of seizures
            bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # predicted
            bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
            
            # plot figures
            if bounds_pred.shape[0] > 0:
            
                # merge seizures close together
                bounds_pred = merge_close(bounds_pred, merge_margin = 5)
            
                # find matching seizures
                detected = match_szrs(bounds_true, bounds_pred, err_margin = 10)
                
                # get number of seizures
                szrs[i,0,ii] = bounds_true.shape[0] # true seizure number
                szrs[i,1,ii] = detected # number of true seizures detected
                szrs[i,2,ii] = bounds_pred.shape[0] - detected # number of extra seizures detected         
                
                # get total numbers
                df.at[ii, 'true_total'] += bounds_true.shape[0]
                df.at[ii, 'total_detected'] += detected
                df.at[ii, 'total_exta'] += (bounds_pred.shape[0] - detected)
   
    return df, szrs


        


tic = time.time() # start timer       
df, szrs =  file_loop(main_path)  
print('Time elapsed = ',time.time() - tic, 'seconds.')       
# df.to_pickle('df_szrs_std_3')
# np.save('szrs_std_3.npy', szrs)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


