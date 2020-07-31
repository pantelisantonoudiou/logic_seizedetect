# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:29:12 2020

@author: Pante
"""

import os, features, time
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_data
from build_feature_data import get_data, get_features_allch
from array_helper import find_szr_idx, match_szrs, merge_close
import matplotlib.pyplot as plt

main_path =  r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'  # 3514_3553_3639_3640  3642_3641_3560_3514
# folder_path = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\3642_3641_3560_3514'
num_channels = [0,1]


# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,)
cross_ch_param_list = () # features.cross_corr


def multi_folder(main_path, thresh_multiplier = 5):
    
    # get subdirectories
    folders = [f.name for f in os.scandir(main_path) if f.is_dir()]
    
    for folder in folders:
        print('Analyzing', folder, '...' )
        
        # get dataframe with detected seizures
        df, szrs =  folder_loop(os.path.join(main_path,folder), thresh_multiplier = thresh_multiplier)
        
        # save dataframe as csv file
        df_path = os.path.join(main_path,folder+'_thresh_'+str(thresh_multiplier) + '_test.csv')
        df.to_csv(df_path, index=True)


def folder_loop(folder_path, thresh_multiplier = 5):
    
    # get file list 
    ver_path = os.path.join(folder_path, 'verified_predictions_pantelis')
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
    filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
    
    # create feature labels
    feature_labels = [x.__name__ + '_1' for x in param_list]; 
    # feature_labels += [x.__name__ + '_2' for x in param_list]; 
    feature_labels += [x.__name__  for x in cross_ch_param_list]
    feature_labels = np.array(feature_labels)
    
    # create dataframe
    columns = ['true_total', 'total_detected', 'total_exta']
    df = pd.DataFrame(data= np.zeros((len(feature_labels),len(columns))), columns = columns, dtype=np.int64)
    df['Features'] = feature_labels
    
    # create seizure array
    szrs = np.zeros((len(filelist),3,feature_labels.shape[0]))
        
    for i in tqdm(range(0, len(filelist))): # loop through experiments  

        # get data and true labels
        data, y_true = get_data(folder_path,filelist[i], ch_num = num_channels, 
                                inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)
        
        ## UNCOMMENT LINE BELOW TO : Clean and filter data
        # data = preprocess_data(data,  clean = True, filt = True, verbose = 0)
        # print('-> data pre-processed.')
        
        # Get features and labels
        x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)
        
        #  UNCOMMENT LINES BELOW TO : get refined data (multiply channels)
        # new_data = np.multiply(x_data[:,0:len(param_list)],x_data[:,len(param_list):x_data.shape[1]-len(cross_ch_param_list)])
        # x_data = np.concatenate((new_data, x_data[:,x_data.shape[1]-1:]), axis=1)
        
        # Normalize data
        x_data = StandardScaler().fit_transform(x_data)

        for ii in range(len(feature_labels)): # iterate through parameteres  x_data.shape[1]

            # get boolean index
            y_pred1 = x_data[:,ii]> (np.mean(x_data[:,ii]) + thresh_multiplier*np.std(x_data[:,ii]))
            y_pred2 = x_data[:,ii+len(feature_labels)]> (np.mean(x_data[:,ii+len(feature_labels)]) + thresh_multiplier*np.std(x_data[:,ii+len(feature_labels)]))
            
            y_pred = (y_pred1.astype(int) + y_pred2.astype(int)) == 2
            ## UNCOMMENT LINE BELOW: for running threshold
            ## y_pred = running_std_detection(x_data[:,ii] , 5, int(60/5)*120)
            
            # get number of seizures
            bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # predicted
            bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
            
            # get true number of seizures
            szrs[i,0,ii] = bounds_true.shape[0] 
            
            # plot figures
            if bounds_pred.shape[0] > 0:
            
                # merge seizures close together
                bounds_pred = merge_close(bounds_pred, merge_margin = 5)
            
                # find matching seizures
                detected = match_szrs(bounds_true, bounds_pred, err_margin = 10)
                
                # get number of matching and extra seizures detected
                szrs[i,1,ii] = detected # number of true seizures detected
                szrs[i,2,ii] = bounds_pred.shape[0] - detected # number of extra seizures detected         
                
            # get total numbers
            df.at[ii, 'true_total'] += szrs[i,0,ii]
            df.at[ii, 'total_detected'] +=  szrs[i,1,ii]
            df.at[ii, 'total_exta'] += szrs[i,2,ii]
   
    return df, szrs


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



if __name__ == '__main__':

    tic = time.time() # start timer       
    multi_folder(main_path, thresh_multiplier = 4)
    print('Time elapsed = ',time.time() - tic, 'seconds.')  

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


