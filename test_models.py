# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:11:04 2020

@author: Pante
"""

##### ---------------------------------------------------- IMPORTS ----------------------------------------------------- #####
import os, sys, features
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from build_feature_data import get_data, get_features_allch
from array_helper import find_szr_idx, match_szrs, merge_close
##### -------------------------------------------------------------------------------------------------------------------- #####

##### ---------------------------------------------------- SETTINGS ----------------------------------------------------- #####
main_path =  r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'  # 3514_3553_3639_3640  3642_3641_3560_3514
ch_list = [0,1] # channel list

# Define total parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,) # single channel features
cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,) # cross channel features
##### -------------------------------------------------------------------------------------------------------------------- #####

def get_feature_parameters(main_path):
    """
    thresh_array, weights, feature_set = get_feature_parameters(main_path)

    Parameters
    ----------
    main_path : Str
    
    Returns
    -------
    thresh_array : List
    weights : List
    feature_set : List

    """
    # Get feature properties
    df = pd.read_csv(os.path.join(main_path,'feature_properties.csv')) # read feature_properties into df
    features = np.array(df.columns[1:]).reshape(1,-1) # get features
    ranks = np.array(df.loc[df['metrics'] == 'ranks'])[:,1:] # get ranks
    optimum_threshold = np.array(df.loc[df['metrics'] == 'optimum_threshold'])[:,1:] # get optimum thresholds for each feature
    
    # Define different threshold levels for testing
    thresh_array =  [optimum_threshold-1, optimum_threshold-0.5, optimum_threshold,
                     optimum_threshold+0.5, optimum_threshold+1, np.ones((1,features.shape[1]))*2,
                     np.ones((1,features.shape[1]))*2.5, np.ones((1,features.shape[1]))*3, np.ones((1,features.shape[1]))*3.5]
    
    # Define two sets of weights
    weights = [np.ones((1,features.shape[1]),dtype=bool), ranks]
    
    # Define feature sets
    feature_set_or = [np.ones((1,features.shape[1]), dtype=bool), ranks>np.percentile(ranks,25),
                    ranks>np.percentile(ranks,50), ranks>np.percentile(ranks,75)]
    
    # Expand feature dataset by randomly dropping 1/3 of True features
    n_repeat = 5 # number of times to drop features per dataset
    feature_set = feature_set_or.copy()
    for i in range(len(feature_set_or)):
        for ii in range(n_repeat):
            temp_feature = feature_set_or[i].copy() # get a copy
            true_idx = np.where(temp_feature)[1] # get true index
            idx = np.random.choice(true_idx, np.int(len(true_idx)/2), replace=True) # get idx to convert to False
            temp_feature[0, idx] = False # convert to false
            feature_set.append(temp_feature)
            
    return thresh_array, weights, feature_set
        

class MethodTest:
    """ MethodTest
    Tests different feature combinations for seizure prediction
    """
    
    # class constructor (data retrieval)
    def __init__(self, main_path):
        """
        ThreshMetrics(main_path)

        Parameters
        ----------
        input_path : Str, path to parent directory.
        """
        
        # pass parameters to object
        self.main_path = main_path # main path
        self.ch_list = ch_list # chanel list to be analyzed

        # create feature labels
        self.feature_labels=[]
        for n in ch_list:
            self.feature_labels += [x.__name__ + '_'+ str(n) for x in param_list]
        self.feature_labels += [x.__name__  for x in cross_ch_param_list]
        self.feature_labels = np.array(self.feature_labels)
        
        # get feature parameters for method testing
        self.thresh_array, self.weights, self.feature_set = get_feature_parameters(Path(main_path).parents[0])
        
        # define metrics
        self.metrics = ['total', 'detected', 'detected_ratio', 'false_positives']
        
    def multi_folder(self):
        """
        multi_folder(self)
        Loop though folder paths get seizure metrics and save to csv
    
        Parameters
        ----------
        main_path : Str, to parent dir
    
        """
        print('--------------------- START --------------------------')
        print('Testing methods on :', self.main_path)
        
        # get save dir
        self.save_folder = os.path.join(self.main_path, 'model_performance') 
        if os.path.exists(self.save_folder) is False:  # create save folder if it doesn't exist
            os.mkdir(self.save_folder)
        
        # create df columns
        self.columns = self.metrics+ ['Enabled_' + x for x in self.feature_labels] \
        + ['Weight_' + x for x in self.feature_labels] + ['Thresh_' + x for x in self.feature_labels]
        
        # create empty df
        self.df = pd.DataFrame(data= np.zeros((0, len(self.columns))), columns = self.columns, dtype=np.int64)
        
        # get subdirectories
        folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]
    
        for i in range(len(folders)): # iterate through folders
            print('Analyzing', folders[i], '...' )
            
            # append seizure properties to dataframe from folder
            # self.folder_loop(folders[i])
        
        # get detected ratio
        self.df['detected_ratio'] = self.df['detected']/self.df['total']
        
        # save dataframe to csv
        file_name = os.path.join(self.save_folder, 'threshold_'+ str(self.threshold).replace('.', '-') +'.csv')
        self.df.to_csv(file_name, header=True, index = False)
        print('Method metrics saved to:', file_name)
        print('----------------------- END --------------------------')


    def folder_loop(self, folder_name):
        """
        folder_loop(self, folder_name)

        Parameters
        ----------
        folder_name : Str, folder name

        Returns
        -------
        bool
        """
        
        # get file list 
        ver_path = os.path.join(self.main_path, folder_name,'verified_predictions_pantelis')
        if os.path.exists(ver_path)== False: # error check
                print('path not found, skipping:', os.path.join(self.main_path, folder_name) ,'.')
                return False
        filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
        filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
     
        for i in tqdm(range(0, len(filelist))): # iterate through experiments
    
            # get data and true labels
            data, y_true = get_data(os.path.join(self.main_path, folder_name),filelist[i], ch_num = ch_list, 
                                    inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)
            
            x_data, labels = get_features_allch(data,param_list,cross_ch_param_list) # Get features and labels
            x_data = StandardScaler().fit_transform(x_data) # Normalize data
            bounds_true = find_szr_idx(y_true, np.array([0,1])) # get bounds of true seizures
            
            for ii in range(len(self.thresh_array)):
                # detect seizures bigger than threshold
                y_pred_array = x_data[:,ii]> (np.mean(x_data[:,ii]) + self.thresh_array[ii]*np.std(x_data[:,ii]))
                self.append_pred(y_pred_array, self.thresh_array[ii]) # append predictions
        return True
    
    
    def append_pred(self, y_pred_array, thresh):
        
        for i in range(len(self.weights)):        
            for ii in range(len(self.feature_set)):      
                
                temp_df = pd.DataFrame(data= np.zeros((0, len(self.columns))), columns = self.columns, dtype=np.int64)
                df1.append(temp_df, ignore_index = True)
                
                y_pred = y_pred_array * self.weights[i] * self.feature_set[ii]
                bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # total predicted
                
                detected =0
                if bounds_pred.shape[0] > 0:
                    # get bounds of predicted sezures
                    bounds_pred = merge_close(bounds_pred, merge_margin = 5) # merge seizures close together                  
                    detected = match_szrs(bounds_true, bounds_pred, err_margin = 10) # find matching seizures
                    
                    
                # get total numbers
                temp_df['total'] += bounds_true.shape[0] 
                temp_df['detected'] += detected
                temp_df['false_positives'] += bounds_pred.shape[0] - detected


obj = MethodTest(main_path) 
obj.multi_folder()

# if __name__ == '__main__':
    
#     if len(sys.argv) == 2:
#         obj = ThreshMetrics(sys.argv[1]) # instantiate and pass main path
#         obj.multi_folder() # get catalogue for multiple folders
#     else:
#         print('Please provide parent directory')