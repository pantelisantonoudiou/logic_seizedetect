# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:29:12 2020

@author: Pante
"""

import os, features, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_data
from build_feature_data import get_data, get_features_allch
from array_helper import find_szr_idx, match_szrs, merge_close
import matplotlib.pyplot as plt

main_path = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\3514_3553_3639_3640' # 3514_3553_3639_3640  3642_3641_3560_3514
num_channels = [0,1]

# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,)
cross_ch_param_list = (features.cross_corr,features.signal_covar)

tic = time.time() # start timer

def file_loop(main_path):
    
    # get data list 
    ver_path = os.path.join(main_path, 'verified_predictions_pantelis')
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
    filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
    
    true_total = 0
    total_detected = 0
    total_exta = 0
    for i in range(0,len(filelist)): # loop through files #

        # get data and true labels
        data, y_true = get_data(main_path,filelist[i],ch_num = num_channels)
        print('->',filelist[i], 'loaded.')
        
        # Clean and filter data
        data = preprocess_data(data,  clean = True, filt = False)
        print('-> data pre-processed.')
        
        # Get features and labels
        x_data, feature_labels = get_features_allch(data,param_list,cross_ch_param_list)
        print('-> features extracted')
        
        # Normalize data
        x_data = StandardScaler().fit_transform(x_data)
        
        # make predictions
        xbest = x_data[:,1] * x_data[:,9] 
        threshold = np.mean(xbest) + 4*np.std(xbest)
        y_pred = xbest>threshold      
        breakpoint()
        # get number of  seizures
        bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # predicted
        bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
        
        # plot figures
        if bounds_pred.shape[0] > 0:
            # plt.figure()
            # ax = plt.axes()
            # ax.plot(xbest,c='k')
            # y = xbest 
            # x =  np.linspace(1,y.shape[0],y.shape[0])
            # ix = np.where(y_true == 1)
            # ax.scatter(x[ix], y[ix], c = 'blue', label = 'true', s = 15)
            # ix = np.where(y_pred == 1)
            # ax.scatter(x[ix], y[ix], c = 'orange', label = 'predicted', s = 8)
            # ax.legend()
        
            # merge seizures close together
            bounds_pred = merge_close(bounds_pred, merge_margin = 5)
        
            # find matching seizures
            detected = match_szrs(bounds_true, bounds_pred, err_margin = 10)
        
            print('Detected', detected, 'out of', bounds_true.shape[0], 'seizures') 
            print('+',bounds_pred.shape[0] - detected ,'extra \n' )
            
            true_total +=  bounds_true.shape[0]
            total_detected += detected
            total_exta += bounds_pred.shape[0] - detected
     
    print('Total detected', total_detected, 'out of', true_total, 'seizures')
    print(total_exta, 'extra seizures')
    print('Time elapsed = ',time.time() - tic, 'seconds.')
    return true_total, total_detected, total_exta
       
true_total, total_detected, total_exta =  file_loop(main_path)  
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


