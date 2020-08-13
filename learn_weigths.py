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
import matplotlib.pyplot as plt


def create_cost(bounds_true, bounds_pred):

    # find matching seizurs
    detected = 0
    a = 1
    if bounds_pred.shape[0]>0:
        detected = match_szrs(bounds_true, bounds_pred, err_margin = 10)
    
    if bounds_true.shape[0]>0:
        # get detected ratio
        a = detected/bounds_true.shape[0]
    
    # get false positives
    b = (bounds_pred.shape[0] - detected)
    
    # cost function
    L = 1 # learning rate
    cost = (-(1-a)*2 + np.log10(b+1))
    
    return cost

from sklearn.metrics import log_loss,recall_score







def find_threshold(x_data, y_true):
    
    # thresh = 1;
    ftr = 1
    
    x = x_data[:,ftr]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # t = np.ones(x.shape[0]) * (np.mean(x) + thresh*np.std(x))
    # line1 = ax.plot(x)
    # line2 = ax.plot(t)
    
    n_loop = 1000
    cost_array = np.zeros(n_loop)
    thresh_array = np.zeros(n_loop)
    thresh_array = np.linspace(0, 20, n_loop)
    for i in range(n_loop):
        
        # thresh_array[i] = thresh  
        y_pred = x> (np.mean(x) + thresh_array[i]*np.std(x))
        
        # get number of seizures
        bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
        bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # predicted
        
        # merge seizures close together
        if bounds_pred.shape[0]>1:
            bounds_pred = merge_close(bounds_pred, merge_margin = 5)
        
        cost = create_cost(bounds_true, bounds_pred) # get cost
        
        # cost = log_loss(y_true, y_pred ,labels =[True,False])
        
        cost_array[i] = cost
        
        # if cost == 0:
        #     print('cost has reached zero, stopping')
        #     return cost_array,thresh_array
        # thresh += cost # update cost
        # ax.plot(np.ones(x.shape[0]) * (np.mean(x) + thresh*np.std(x)))
        # line2[0].set_ydata(np.ones(x.shape[0]) * (np.mean(x) + thresh*np.std(x)))
        # fig.canvas.draw()  
    
    plt.figure()        
    plt.plot(thresh_array, cost_array)
    plt.ylabel('cost')
    plt.xlabel('thresh')
    return cost_array,thresh_array
        
        
def find_threshold_all(x_data, y_true):
    
    thresh = 1;
    ftr = 1
    
    x = x_data[:,ftr]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = np.ones(x.shape[0]) * (np.mean(x) + thresh*np.std(x))
    line1 = ax.plot(x)
    line2 = ax.plot(t)
    
    n_loop = 100
    cost_array = np.zeros(n_loop)
    thresh_array = np.zeros(n_loop)
    # thresh_array = np.linspace(10, 0, n_loop)
    for i in range(n_loop):
        
        thresh_array[i] = thresh  
        y_pred = x> (np.mean(x) + thresh_array[i]*np.std(x))
        
        # get number of seizures
        bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
        bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # predicted
        
        # merge seizures close together
        if bounds_pred.shape[0]>1:
            bounds_pred = merge_close(bounds_pred, merge_margin = 5)
        
        cost = create_cost(bounds_true, bounds_pred) # get cost
        
        # cost = log_loss(y_true, y_pred ,labels =[True,False])
        
        cost_array[i] = cost
        
        if cost == 0:
            print('cost has reached zero, stopping')
            return cost_array,thresh_array

    return cost_array,thresh_array
        



# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,)
cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,)


  # get data and true labels
exp_path  = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\3642_3641_3560_3514'
# 071919_3514 071719_3560
data, y_true = get_data(exp_path, '071719_3560',ch_num = [0,1], 
                        inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)

# get features
x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)

# Normalize data
x_data = StandardScaler().fit_transform(x_data)

cost_array,thresh_array = find_threshold(x_data, y_true)




















