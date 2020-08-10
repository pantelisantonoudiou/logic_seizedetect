# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:47:30 2020

@author: Pante
"""


import os, features, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_data
from build_feature_data import get_data, get_features_allch
from array_helper import find_szr_idx


main_path =  r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'  # 3514_3553_3639_3640  3642_3641_3560_3514
# folder_path = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\3642_3641_3560_3514'
num_channels = [0,1]
win = 5


# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,)
cross_ch_param_list = () # features.cross_corr


def multi_folder(main_path):
    
    # get subdirectories
    folders = [f.name for f in os.scandir(main_path) if f.is_dir()]
    
    for i in range(len(folders)):
        print('Analyzing', folders[i], '...' )
        
        # get dataframe with detected seizures
        df_temp = folder_loop(main_path,folders[i])
        
        if i == 0: # create copy of the dataframe
            df = df_temp.copy()
        else: # append to dataframe
            df.append(df_temp)    

    return df


def folder_loop(main_path,folder_path):
    
    # get file list 
    ver_path = os.path.join(main_path,folder_path, 'verified_predictions_pantelis')
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
    filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
    
    # create feature labels
    feature_labels = [x.__name__ + '_1' for x in param_list]; 
    feature_labels += [x.__name__ + '_2' for x in param_list]; 
    feature_labels += [x.__name__  for x in cross_ch_param_list]
    feature_labels = np.array(feature_labels)
    
    # define time bins
    time_bins = np.array([
                         [-120, -90],
                         [-90, -60], 
                         [-60, -30], 
                         [-30, 0],
                         [0, 30],
                         [30, 60],
                         [60, 90],
                         [90, 120], 
                         ])
    
    # define fixed columns
    columns = ['exp_id', 'szr_start', 'szr_end', 'szr_percentile','x_sdevs', 'during_szr']
    
    time_cols = [] # convert time bins to headers
    for x in time_bins.tolist():
        time_cols.append('_'.join(map(str, x))) 
    columns.extend(time_cols)  # extend original columns list   

    for i in tqdm(range(0,len(filelist))): # loop through experiments   len(filelist)

        # get data and true labels
        data, y_true = get_data(os.path.join(main_path,folder_path),filelist[i], ch_num = num_channels, 
                                inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)
        
        ## UNCOMMENT LINE BELOW TO : Clean and filter data
        # data = preprocess_data(data,  clean = True, filt = True, verbose = 0)
        
        # Get features and labels
        x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)
        
        # Normalize data
        x_data = StandardScaler().fit_transform(x_data)
        
        # GET multifeatures????
        for ii in range(1): # iterate through parameteres  x_data.shape[1] len(feature_labels) 
        
            # create dataframe
            df = pd.DataFrame(data= np.zeros((0,len(columns))), columns = columns, dtype=np.int64)

            # get seizure index
            bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
 
            if bounds_true.shape[0] > 0:
        
                # get seizure and surround properties
                szrs = get_surround(x_data[:,ii], bounds_true, time_bins)
                
                # insert seizure start and end
                df['exp_id'] = [os.path.join(folder_path,filelist[i])] * bounds_true.shape[0]
                df['szr_start'] = bounds_true[:,0]
                df['szr_end'] =  bounds_true[:,1]
                
                # append seizure properties
                df.iloc[:, 3:] = szrs 
                
                # append to dataframe
                df.to_csv(labels[i] +'.csv', mode='a', header=True, index = False) # ADD CORRECT NAME
    return df

# @jit(nopython = True)
from scipy.stats import percentileofscore
def get_surround(feature, idx, time_bins):
    """
    get_surround(feature, idx, time_bins)

    Parameters
    ----------
    feature : TYPE
    idx : TYPE
    surround_time : TYPE

    Returns
    -------
    szrs : 2d ndarray (rows = seizures, cols = features)

    """
    
    # convert time bins to window bins
    time_bins = time_bins/win
    time_bins = time_bins.astype(np.int64) # convert to integer
    
    # create empty vectors to store before, after and within seizure features
    fixed_bins = 3
    szrs = np.zeros((idx.shape[0],len(time_bins) + fixed_bins))
    
    for i in range(idx.shape[0]): # iterate over seizure number
        
        szr_feature = np.mean(feature[idx[i,0]:idx[i,1]]) # during
        szrs[i,0] = percentileofscore(feature, szr_feature) # get percentile
        szrs[i,1] = (szr_feature-np.mean(feature))/np.std(feature) # get deviations from mean  
        szrs[i,2] = szr_feature
        
        for ii in range(time_bins.shape[0]): # get mean of each time bin
            if time_bins[ii,0] < 0:
                idx_start =  idx[i,0] + time_bins[ii,0];
                idx_end = idx[i,0] + time_bins[ii,1];
            else:
                idx_start =  idx[i,1] + time_bins[ii,0];
                idx_end = idx[i,1] + time_bins[ii,1];
            print(idx_start, idx_end)
            
            szrs[i,ii+fixed_bins] = np.mean(feature[idx_start : idx_end])
            
    return szrs

@jit(nopython = True)
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#  remapped_array = remap_array(np.sort(feature)) # remap array from 0 to 100
@jit(nopython = True)
def remap_array(array):
    min_value = array.min()
    max_value = array.max()
    a = (array - min_value) / (max_value - min_value) * 100
    return a


if __name__ == '__main__':

    tic = time.time() # start timer       
    df = multi_folder(main_path)
    print('Time elapsed = ',time.time() - tic, 'seconds.')  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    