# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:14:32 2020

@author: Pante
"""
import os, features, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from array_helper import find_szr_idx, match_szrs, merge_close
from build_feature_data import get_data, get_features_allch


# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,)
cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,)


 # get data and true labels
exp_path  = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\3642_3641_3560_3514'
data, y_true = get_data(exp_path, '071919_3560',ch_num = [0,1], 
                        inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)

# get features
x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)


# Normalize data
x_data = StandardScaler().fit_transform(x_data)


def create_cost(y_true, y_pred):
    # get number of seizures
    bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
    bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # predicted
    
    # merge seizures close together
    bounds_pred = merge_close(bounds_pred, merge_margin = 5)
    
    # find matching seizurs
    detected = match_szrs(bounds_true, bounds_pred, err_margin = 10)
    
    # get detected ratio
    a = detected/bounds_true.shape[0]
    
    # get false positive ratio
    b = (bounds_pred.shape[0] - detected)
    
    # cost function
    cost = a + b/100
    
    return cost



def find_threshold(x, y_true ):
    
    thresh_init = 0;
    ftr = 1
    for i in range(100):
        y_pred = x[:,ftr]> (np.mean(x[:,ftr]) + thresh_init*np.std(x[:,ftr]))
        
        cost = create_cost(y_true, y_pred)
        thresh_init -= cost


























