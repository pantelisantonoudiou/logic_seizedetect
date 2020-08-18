# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:12:09 2020

@author: Pante
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:29:12 2020

@author: Pante
"""

import os, features, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from build_feature_data import get_data, get_features_allch
from array_helper import find_szr_idx, match_szrs, merge_close

### -------------SETTINGS --------------###
# main_path =  r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'  # 3514_3553_3639_3640  3642_3641_3560_3514
ch_list = [0,1] # channel list

# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,) # single channel features
cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,) # cross channel features
##### ----------------------------------------------------------------- #####

class ThreshMetrics:
    """ ThreshMetrics
    Retreive detection score and false positives across thresholds
    """
    
    # class constructor (data retrieval)
    def __init__(self, main_path):
        """
        GetCatalogue(main_path)

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
        

    def multi_folder(self):
        """
        multi_folder(self)
        Loop though folder paths and append seizure properties to csv
    
        Parameters
        ----------
        main_path : Str, to parent dir
    
        """
        print('------------------------------------------------------')
        print('------------------------------------------------------')
        print('Creating seizure catalogue from:', self.main_path,'\n')
        
        # get save dir
        self.save_folder = os.path.join(self.main_path, 'optimum_threshold')
        
        # create save folder
        if os.path.exists(self.save_folder) is False:
            os.mkdir(self.save_folder)
            
        # create csv file for each parameter
        metrics = ['total', 'detected', 'detected_ratio','false_positives']
        df = pd.DataFrame(data= np.zeros((len(metrics), len(self.feature_labels))), columns = self.feature_labels, dtype=np.int64)
        df['metrics'] = metrics
            
        # get subdirectories
        folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]
    
        for i in range(len(folders)): # iterate through folders
            print('Analyzing', folders[i], '...' )
            
            # append seizure properties to dataframe from folder
            self.folder_loop(folders[i])
            
        df.to_csv(os.path.join(self.save_folder, 'threshold!!!!!' +'.csv'), mode='a', header=True, index = False)
        print('Seizure catalogue successfully created.')
        print('------------------------------------------------------')
        print('------------------------------------------------------')


def folder_loop(self, folder_path):
    
    # get file list 
    ver_path = os.path.join(folder_path, 'verified_predictions_pantelis')
    if os.path.exists(ver_path)== False: # error check
            print('path not found, skipping:', os.path.join(self.main_path, folder_path) ,'.')
            return False
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
    filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
     
    # create dataframe
    columns = ['true_total', 'total_detected', 'total_exta']
    df = pd.DataFrame(data= np.zeros((len(feature_labels),len(columns))), columns = columns, dtype=np.int64)
    
    # create seizure array
    szrs = np.zeros((len(filelist),3,feature_labels.shape[0]))
    
 
    for i in tqdm(range(0, len(filelist))): # loop through experiments

        # get data and true labels
        data, y_true = get_data(folder_path,filelist[i], ch_num = ch_list, 
                                inner_path={'data_path':'filt_data', 'pred_path':'verified_predictions_pantelis'} , load_y = True)
        
        # Get features and labels
        x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)

        # Normalize data
        x_data = StandardScaler().fit_transform(x_data)
    
        for ii in range(len(feature_labels)): # iterate through parameteres  x_data.shape[1]

            # SD
            y_pred = x_data[:,ii]> (np.mean(x_data[:,ii]) + thresh_multiplier*np.std(x_data[:,ii]))
            
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





if __name__ == '__main__':

    tic = time.time() # start timer       
    multi_folder(main_path, thresh_multiplier = 3)
    print('Time elapsed = ',time.time() - tic, 'seconds.')  

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


