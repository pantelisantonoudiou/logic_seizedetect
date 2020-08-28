# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:03:00 2020

@author: Pante
"""

import os, sys, features
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from build_feature_data import get_data, get_features_allch
from scipy.stats import pearsonr
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

### -------------SETTINGS --------------###
# parent_path =  r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'  # 3514_3553_3639_3640  3642_3641_3560_3514
ch_list = [0,1] # channel list

# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
              features.get_envelope_max_diff,) # single channel features
cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,) # cross channel features
##### ----------------------------------------------------------------- #####

class FeatureSelection:
    """ FeatureSelection
    Get scores for different features and save to csv
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
        
        # define metrics
        self.metrics = [pearson_corr, anova_f, random_forest]

    def multi_folder(self):
        """
        multi_folder(self)
        Loop though folder paths get feature metrics and save to csv
    
        Parameters
        ----------
        main_path : Str, to parent dir
    
        """
        print('--------------------- START --------------------------')
        print('Getting feature metrics from :', self.main_path)
        
        # get save dir
        self.save_folder = os.path.join(self.main_path, 'feature_selection') 
        if os.path.exists(self.save_folder) is False:  # create save folder if it doesn't exist
            os.mkdir(self.save_folder)
                
        # create csv file for each metric
        for ii in range(len(self.metrics)): # iterate through parameteres
            # create empty dataframe
            df = pd.DataFrame(data = np.zeros((0, len(self.feature_labels))), columns = self.feature_labels)
            df.insert(loc = 0, column = 'exp_id', value = '')
            df.to_csv(os.path.join(self.save_folder, self.metrics[ii].__name__ +'.csv'), mode='a', header=True, index = False)     

            # get subdirectories
            folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]
        
        for i in range(len(folders)): # iterate through folders
            print('Analyzing', folders[i], '...' )
            
            # append seizure properties to dataframe from folder
            self.folder_loop(folders[i])
          
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
            if np.sum(y_true)>3:
                # Get features and labels
                x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)
                x_data = StandardScaler().fit_transform(x_data) # Normalize data
                
                for ii in range(len(self.metrics)):
                    # get metric for features
                    metric = self.metrics[ii](x_data, y_true)
                    
                    # create dateframe with metric
                    df = pd.DataFrame(data = np.array(metric).reshape(1,-1), columns = self.feature_labels)
                    df.insert(loc = 0, column = 'exp_id', value = filelist[i])
                    
                    # save to csv
                    df.to_csv(os.path.join(self.save_folder, self.metrics[ii].__name__ +'.csv'), mode='a', header=False, index = False)
        return True
    

### ----------------- GET METRICS TO SELECT FEATURES  --------------------------- ###
# get pearson correlation value    
def pearson_corr(x_data, y_true):
    """
    pearson_corr(x_data, y_true)

    Parameters
    ----------
    x_data :  2D ndarray (rows = segments, columns = features)
    y_true : 1D ndarray, bool, ground truth labels

    Returns
    -------
    r2_vals : 1D ndarray, pearson correlation coefficients for each feature
    p_vals : 1D ndarray, p-value for each feature

    """

    # create arrays for storage
    r2_vals = np.zeros(x_data.shape[1]) # pearson r
    p_vals = np.zeros(x_data.shape[1]) # p-value
    
    for i in range(x_data.shape[1]):
        (r2_vals[i], p_vals[i])  = pearsonr(x_data[:,i], y_true)
        
    return r2_vals

# get anova f values
def anova_f(x_data, y_true):
    """
    anova_f(x_data, y_true)

    Parameters
    ----------
    x_data :  2D ndarray (rows = segments, columns = features)
    y_true : 1D ndarray, bool, ground truth labels

    Returns
    -------
    f_vals : 1D ndarray, anova f values
    """
    
    f_vals, p = f_classif(x_data, y_true)
    return f_vals
   
# get Random forest importances
def random_forest(x_data, y_true):
    """
    random_forest(x_data, y_true)

    Parameters
    ----------
    x_data :  2D ndarray (rows = segments, columns = features)
    y_true : 1D ndarray, bool, ground truth labels

    Returns
    -------
    feature_importances_ : 1D ndarray, ranks of features -> 1-0 (highest-lowest)
    """
    
    model = RandomForestClassifier(n_estimators = 100)
    model.fit(x_data, y_true)
    return model.feature_importances_ 

# Log loss feature drop, need to change df and 
def log_loss_feature_drop(x_data, y_true, labels):
    """
    test_model(x_data, y_true, labels)
    
    Parameters
    ----------
    x_data : TYPE
    y_true : TYPE
    labels : TYPE

    Returns
    -------
    df_counts : dataframe (columns= features), normalized counts for kept features
    """
    
    optimum_threshold = 3; #[2,3,4,5]
    
    repeats = 10; # number of repeats
    
    # create counts dataframe
    df_counts = pd.DataFrame(data = np.zeros((1, len(labels))), columns = labels)
    
    # get predictions for each feature
    y_pred_array = x_data > (np.std(x_data, axis=0) * optimum_threshold)
    df = pd.DataFrame(data = y_pred_array, columns = labels)
    
    # calculate initial cost
    # current_cost = user_cost(y_true, np.mean(y_pred_array, axis=1) > 0.5)
    current_cost =  log_loss(y_true, np.mean(y_pred_array, axis=1) > 0.5)
   
    for ii in range(repeats): # repeat for robust findings
    
        # get random drop list sequence
        drop_list = labels[np.random.permutation(x_data.shape[1])]
            
        y_pred_temp = df  # get original dataframe
        
        for i in range(x_data.shape[1]): # loop through features
             
            y_pred = np.mean(y_pred_temp.drop(drop_list[i], axis=1) , axis=1) > 0.5 # get predecitions
            # cost = user_cost(y_true, np.array(y_pred))  # calculate cost 
            cost = log_loss(y_true, y_pred)

            if cost <= current_cost:
                y_pred = y_pred_temp.drop(drop_list[i], axis=1)  # drop idx from prediction array
            else:
                cost = current_cost; # update cost
                df_counts[drop_list[i]] +=1 # add 1 if the feature is kept
                
    
    df_counts/=repeats # normalize
    df_counts = df_counts > 0.5 # get maximum vote
    return df_counts  

### --------------------------------------------------------------------- ###


if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        obj = FeatureSelection(sys.argv[1]) # instantiate and pass main path 
        obj.multi_folder() # get catalogue for multiple folders
    else:
        print('Please provide parent directory')
        
        
        
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        


