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
from sklearn.metrics import log_loss,recall_score
import matplotlib.pyplot as plt


####### consider isolation forest for outlier detection!!!!!!

def create_cost(bounds_true, bounds_pred):
    """
    create_cost(bounds_true, bounds_pred)

    Parameters
    ----------
    bounds_true : 2d ndarray (rows = seizrs, columns = start,stop), ground truth
    bounds_pred : 2d ndarray (rows = seizrs, columns = start,stop), predicted

    Returns
    -------
    cost : Float,

    """

    # find matching seizurs
    detected = 0
    a = 100
    if bounds_pred.shape[0]>0:
        detected = match_szrs(bounds_true, bounds_pred, err_margin = 10)
    
    if bounds_true.shape[0]>0:
        # get detected ratio
        a = (1 - (detected/bounds_true.shape[0]))*20
        
    # get false positives
    b = (bounds_pred.shape[0] - detected)
    
    # cost function
    # L = 1 # learning rate
    cost = a + np.log10(b+1)
    
    return cost


def szr_cost(bounds_true, bounds_pred):
    """
    create_cost(bounds_true, bounds_pred)

    Parameters
    ----------
    bounds_true : 2d ndarray (rows = seizrs, columns = start,stop), ground truth
    bounds_pred : 2d ndarray (rows = seizrs, columns = start,stop), predicted

    Returns
    -------
    cost : Float,

    """

    # find matching seizurs
    detected = 0
    if bounds_pred.shape[0]>0:
        detected = match_szrs(bounds_true, bounds_pred, err_margin = 10)
    
    if bounds_true.shape[0]>0:
        # get detected ratio
        a = 1 - (detected/bounds_true.shape[0])
    
    if (a > 0 and a <= 1):
        a = 20
        
    # get false positives
    b = (bounds_pred.shape[0] - detected)
    
    # cost function
    cost = a + np.log10(b+1)
    
    return cost


def get_min_cost(feature, y_true):
    """
    get_min_cost(feature, y_true)

    Parameters
    ----------
    feature : 1D ndarray, extracted feature
    y_true : 1D ndarray, bool grund truth labels
    Returns
    -------
    TYPE: Float, threshold value that gves minimum cost

    """

    n_loop = 100 # loop number and separation
    thresh_array = np.linspace(1, 20, n_loop) # thresholds to test
    cost_array = np.zeros(n_loop)
    
    for i in range(n_loop):
        
        # thresh_array[i] = thresh  
        y_pred = feature> (np.mean(feature) + thresh_array[i]*np.std(feature))
        
        # get number of seizures
        bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
        bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # predicted
        
        # merge seizures close together
        if bounds_pred.shape[0]>1:
            bounds_pred = merge_close(bounds_pred, merge_margin = 5)
        
        cost = szr_cost(bounds_true, bounds_pred) # get cost
        
        # pass to array
        cost_array[i] = cost
        
    return thresh_array[np.argmin(cost_array)]



def find_threshold(x_data, y_true):
    
    # thresh = 1;
    ftr = 8
    
    x = x_data[:,ftr]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # t = np.ones(x.shape[0]) * (np.mean(x) + thresh*np.std(x))
    # line1 = ax.plot(x)
    # line2 = ax.plot(t)
    
    n_loop = 100
    cost_array = np.zeros(n_loop)
    thresh_array = np.zeros(n_loop)
    thresh_array = np.linspace(1, 20, n_loop)
    for i in range(n_loop):
        
        # thresh_array[i] = thresh  
        y_pred = x> (np.mean(x) + thresh_array[i]*np.std(x))
        
        # get number of seizures
        bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
        bounds_pred = find_szr_idx(y_pred, np.array([0,2])) # predicted
        
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
    print('seizures = ', bounds_true.shape[0])
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
data, y_true = get_data(exp_path, '071919_3514a',ch_num = [0,1], 
                        inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)


# get file list
main_path = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'
folder_path =  '3514_3553_3639_3640'
ver_path = os.path.join(main_path,folder_path, 'verified_predictions_pantelis')
filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending


# data, y_true = get_data(r'W:\Maguire Lab\Trina\2019\07-July\3514_3553_3639_3640, '071819_3553a',ch_num = [0,1], 
#                         inner_path={'data_path':'reorganized_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)

for i in range(1):
    # 071919_3514 071719_3560
    data, y_true = get_data(os.path.join(main_path, folder_path), filelist[i],ch_num = [0,1], 
                            inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)
    
    if sum(y_true) == 0:
        continue
    
    # get features
    x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)

    # Normalize data
    x_data = StandardScaler().fit_transform(x_data)
    
    # get cost plot
    cost_array,thresh_array = find_threshold(x_data, y_true)




















