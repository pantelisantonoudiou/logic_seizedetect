# -*- coding: utf-8 -*-

##### ---------------------------------------------------- IMPORTS ----------------------------------------------------- #####
import os
# import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from helper import features
from helper.io_getfeatures import get_data, get_features_allch
from helper.array_helper import find_szr_idx, match_szrs, merge_close, match_szrs_idx
##### -------------------------------------------------------------------------------------------------------------------- #####

##### ---------------------------------------------------- SETTINGS ----------------------------------------------------- #####
ch_list = [0,1] # channel list

# Define total parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,) # single channel features
cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,) # cross channel features
##### -------------------------------------------------------------------------------------------------------------------- #####

class MethodTest:
    """
    Tests different feature combinations for seizure prediction
    obtained from testing dataset
    """
    
    # class constructor (data retrieval)
    def __init__(self, main_path):
        """
        
        Parameters
        ----------
        main_path : Str, path to parent directory.
        
        --- Examples ---
        main_path = 'test\data\test'
        obj = MethodTest(main_path)
        obj.multi_folder()
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
        
        # get df from training and empty metrics
        self.df = pd.read_csv(os.path.join(Path(main_path).parents[0], 'all_method_metrics_train.csv'))
        self.df[['total','detected','detected_ratio','false_positives']] = 0
        
        # get parameter index
        self.thresh = np.where(np.array(self.df.columns.str.contains('Thresh')))[0]
        self.weights = np.where(np.array(self.df.columns.str.contains('Weight')))[0]
        self.enabled = np.where(np.array(self.df.columns.str.contains('Enabled')))[0]

        if self.df.columns.str.contains('Enabled').sum() != len(self.feature_labels): 
            print('Error! Size of features from csv file does not match object feature length')
            ## ADD statement to check that all features match!!!
            return
        
        
    def multi_folder(self):
        """
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
        
        # get subdirectories
        folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]

        for i in range(len(folders)): # iterate through folders
            print('Analyzing', folders[i], '...' )
            
            # append seizure properties to dataframe from folder
            self.folder_loop(folders[i])
        
        # get detected ratio
        self.df['detected_ratio'] = self.df['detected']/self.df['total']
        
        # save dataframe to csv
        file_name = os.path.join(self.save_folder, 'all_method_metrics_idx_method.csv')
        self.df.to_csv(file_name, header=True, index=False)
        print('Method metrics saved to:', file_name)
        print('----------------------- END --------------------------')
        

    def folder_loop(self, folder_name):
        """

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
            
            for ii in range(len(self.df)): # iterate through df
                # detect seizures bigger than threshold
                thresh = (np.mean(x_data) + np.array(self.df.loc[ii][self.thresh]) * np.std(x_data)) # get threshold
                y_pred_array = x_data > thresh # get predictions
                
                # find predicted seizures
                w = np.array(self.df.loc[ii][self.weights]) # get weights 
                e =  np.array(self.df.loc[ii][self.enabled]) # get enabled features
                y_pred = y_pred_array * w * e # get predictions based on weights and selected features
                y_pred = np.sum(y_pred, axis=1) / np.sum(w * e) # normalize to weights and selected features
                y_pred = y_pred > 0.5 # get popular vote
                bounds_pred = find_szr_idx(y_pred, np.array([0,1])) # get predicted seizure index
                
                detected = 0 # set default detected to 0
                if bounds_pred.shape[0] > 0:
                    # # get bounds of predicted sezures
                    # bounds_pred = merge_close(bounds_pred, merge_margin = 5) # merge seizures close together                  
                    # detected = match_szrs(bounds_true, bounds_pred, err_margin = 10) # find matching seizures
                    detected = np.sum(match_szrs_idx(bounds_true, y_pred))
                    
                # get total numbers
                self.df.at[ii, 'total'] = self.df['total'][ii] + bounds_true.shape[0]                                       # total true
                self.df.at[ii, 'detected'] = self.df['detected'][ii] + detected                                             # n of detected seizures
                self.df.at[ii, 'false_positives'] = self.df['false_positives'][ii] + (bounds_pred.shape[0] - detected)      # n of false positives
                
        return True


