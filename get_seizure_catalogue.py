# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:47:30 2020

@author: Pante
"""

##### ------------------------------ IMPORTS -------------------------- #####
import os, features, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_data
from scipy.stats import percentileofscore
from build_feature_data import get_data, get_features_allch
from array_helper import find_szr_idx
##### ----------------------------------------------------------------- #####


### -------------SETTINGS --------------###
# main_path =  r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'  # 3514_3553_3639_3640  3642_3641_3560_3514
ch_list = [0,1] # channel list
win = 5 # window duration

# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,)
cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,)

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
##### ----------------------------------------------------------------- #####

class GetCatalogue:
    """
    Class -> GetCatalogue
    Retrieves seizure properties and saves to organized csv for each feature specified
    in param_list and cross_ch_param_list.
    """
    
     # class constructor (data retrieval)
    def __init__(self, main_path):
        """
        GetCatalogue(main_path)

        Parameters
        ----------
        input_path : STRING, Raw data path.
        """
        
        # pass parameters to object
        self.main_path = main_path # main path
        self.ch_list = ch_list # chanel list to be analyzed
        self.win = win # window size (seconds)
        self.time_bins = time_bins # get time bins

        # create feature labels
        self.feature_labels=[]
        for n in ch_list:
            self.feature_labels += [x.__name__ + '_'+ str(n) for x in param_list]
        self.feature_labels += [x.__name__  for x in cross_ch_param_list]
        self.feature_labels = np.array(self.feature_labels)
        
        # define fixed columns
        self.columns = ['exp_id', 'szr_start', 'szr_end', 'szr_percentile','x_sdevs', 'during_szr']
        
        # add time bins to columns
        time_cols = [] # convert time bins to headers
        for x in self.time_bins.tolist():
            time_cols.append('_'.join(map(str, x))) 
        self.columns.extend(time_cols)  # extend original columns list   
        

    def multi_folder(self):
        """
        multi_folder(self)
        Loop though folder paths and append seizure properties to csv
    
        Parameters
        ----------
        main_path : Str, to parent dir
    
        """
        
        # get save dir
        self.save_folder = os.path.join(self.main_path, 'szr_catalogue')
        
        # create save folder
        if os.path.exists(self.save_folder) is False:
            os.mkdir(self.save_folder)
            
        # create csv file for each parameter
        for x in range(len(self.feature_labels)): # iterate through parameteres
           # create empty dataframe
           df = pd.DataFrame(data= np.zeros((0,len(self.columns))), columns = self.columns, dtype=np.int64)
           df.to_csv(os.path.join(self.save_folder, self.feature_labels[x] +'.csv'), mode='a', header=True, index = False)     
            
        # get subdirectories
        folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]
    
        for i in range(len(folders)): # iterate through folders
            print('Analyzing', folders[i], '...' )
            
            # append seizure properties to dataframe from folder
            GetCatalogue.folder_loop(folders[i])
        print('Seizure catalogue successfully created.')
    
    
    def folder_loop(self, folder_path):
        """
        folder_loop(self, folder_path)
        
        Parameters
        ----------
        folder_path : Str, to child dir
    
        """
        
        # get path
        ver_path = os.path.join(self.main_path,folder_path, 'verified_predictions_pantelis')
        if os.path.exists(ver_path)== False:
            print('path not found, skiiping..', self.main_path)
            return False
        
         # get file list
        filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
        filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
            
        
        for i in tqdm(range(0,len(filelist))): # loop through experiments   len(filelist)
    
            # get data and true labels
            data, y_true = get_data(os.path.join(self.main_path, folder_path),filelist[i], ch_num = ch_list, 
                                    inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)
            
            ## UNCOMMENT LINE BELOW TO : Clean and filter data
            # data = preprocess_data(data,  clean = True, filt = True, verbose = 0)
            
            # Get features and labels
            x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)
            
            # Normalize data
            x_data = StandardScaler().fit_transform(x_data)
            
            for ii in range(len(self.feature_labels)): # iterate through parameteres  x_data.shape[1] len(feature_labels) 
            
                # create dataframe
                df = pd.DataFrame(data= np.zeros((0,len(self.columns))), columns = self.columns, dtype=np.int64)
    
                # get seizure index
                bounds_true = find_szr_idx(y_true, np.array([0,1])) # true
     
                if bounds_true.shape[0] > 0:
                    # get seizure and surround properties
                    szrs = GetCatalogue.get_surround(x_data[:,ii], bounds_true, self.time_bins)
                    
                    # insert seizure start and end
                    df['exp_id'] = [os.path.join(folder_path, filelist[i])] * bounds_true.shape[0]
                    df['szr_start'] = bounds_true[:,0]
                    df['szr_end'] =  bounds_true[:,1]
                    
                    # append seizure properties
                    df.iloc[:, 3:] = szrs 
                    
                    # append to dataframe
                    df.to_csv(os.path.join(self.save_folder, self.feature_labels[ii] +'.csv'), mode='a', header=False, index = False)
    
    @staticmethod  
    def get_surround(feature, idx, time_bins):
        """
       GetCatalogue.get_surround(feature, idx, time_bins)
    
        Parameters
        ----------
        feature : 1d ndarray
        idx : 2d ndarray, szr index
        time_bins : 2d ndarray, relative time to seizure to extract properties
    
        Returns
        -------
        szrs : 2d ndarray (rows = seizures, cols = features), seizure properties
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
                
                # add feature mean
                szrs[i,ii+fixed_bins] = np.mean(feature[idx_start : idx_end])
        return szrs
    
    
if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        print('Creating seizure catalogue from', sys.argv[1])
        obj = GetCatalogue(sys.argv[1]) # instantiate and pass main path
        breakpoint()
        obj.multi_folder() # get catalogue for multiple folders
    else:
        print('Please provide parent directory')

    
    



    
    
    
    
    
    
    
    
    
    
    