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
from sklearn.feature_selection import f_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

### -------------SETTINGS --------------###
# main_path =  r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'  # 3514_3553_3639_3640  3642_3641_3560_3514
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
        self.metrics = ['Pearson Corr.', 'ANOVA', 'RFE','Permutation']

    def multi_folder(self):
        """
        multi_folder(self)
        Loop though folder paths get seizure metrics and save to csv
    
        Parameters
        ----------
        main_path : Str, to parent dir
    
        """
        print('--------------------- START --------------------------')
        print('------------------------------------------------------')
        print('Getting metrics from :', self.main_path)
        
        
        # get save dir
        self.save_folder = os.path.join(self.main_path, 'optimum_threshold') 
        if os.path.exists(self.save_folder) is False:  # create save folder if it doesn't exist
            os.mkdir(self.save_folder)
                

        
        for ii in range(len(self.metrics)): # iterate though thresholds
            
            self.threshold = threshold_array[ii] # set threshold
            print('Seizure threshold set at :', self.threshold,'\n')
        
            # create csv file for each parameter
            
            self.df = pd.DataFrame(data= np.zeros((len(self.feature_labels), len(self.metrics))), columns = self.metrics, dtype=np.int64)
            self.df.insert(loc = 0, column ='features', value = self.feature_labels)
                
            # get subdirectories
            folders = [f.name for f in os.scandir(self.main_path) if f.is_dir()]
        
            for i in range(len(folders)): # iterate through folders
                print('Analyzing', folders[i], '...' )
                
                # append seizure properties to dataframe from folder
                self.folder_loop(folders[i])
            
            # get detected ratio
            self.df['detected_ratio'] = self.df['detected']/self.df['total'] 
            # save dataframe to csv
            file_name = os.path.join(self.save_folder, 'threshold_'+ str(self.threshold).replace('.', '-') +'.csv')
            self.df.to_csv(file_name, header=True, index = False)
            print('Seizure metrics saved to:', file_name)
        
        
        print('----------------------- END --------------------------')
        print('------------------------------------------------------')


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
            
            # Get features and labels
            x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)
    
            # Normalize data
            x_data = StandardScaler().fit_transform(x_data)
            

        return True
    

# get pearson correlation value    
def get_correlation(x_data, y_true):
    """
    get_correlation(x_data, y_true)

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
        
    return r2_vals, p_vals


# get ANOVA f values
def get_fvalues(x_data, y_true):
    """
    get_fvalues(x_data, y_true)

    Parameters
    ----------
    x_data :  2D ndarray (rows = segments, columns = features)
    y_true : 1D ndarray, bool, ground truth labels

    Returns
    -------
    f_vals : 1D ndarray, ANOVA f-values for each feature
    """
    
    # create arrays for storage
    f_vals = np.zeros(x_data.shape[1]) # anova f-values
    
    for i in range(x_data.shape[1]):     
        (f_vals[i], p) = f_classif(x_data[:,i], y_true) # perform anova 
    return f_vals
   
# get Recursive feature elimination values
def get_RFE_ranks(x_data, y_true):
    """
    get_RFE_ranks(x_data, y_true)

    Parameters
    ----------
    x_data :  2D ndarray (rows = segments, columns = features)
    y_true : 1D ndarray, bool, ground truth labels

    Returns
    -------
    rfe_ranks : 1D ndarray, ranks of features -> 1-0 (highest-lowest)
    """
    
     # create arrays for storage
    rfe_ranks = np.zeros(x_data.shape[1]) # anova f-values
    
    # Recursive feature elimination
    for i in range(x_data.shape[1]):  
        # init object
        rfe = RFE(estimator=LogisticRegression(), n_features_to_select=1)
        rfe.fit(x_data[:,i], y_true) # fit data
        rfe_ranks[i] =  1/rfe.ranking_ # get ranks

# get Permutation importances
def get_permutation_importances(x_data, y_true):
    """
    get_permutation_importances(x_data, y_true)

    Parameters
    ----------
    x_data :  2D ndarray (rows = segments, columns = features)
    y_true : 1D ndarray, bool, ground truth labels

    Returns
    -------
    rfe_ranks : 1D ndarray, ranks of features -> 1-0 (highest-lowest)
    """
    
     # create arrays for storage
    importances = np.zeros(x_data.shape[1]) # anova f-values
    
    # Recursive feature elimination
    for i in range(x_data.shape[1]):  
        
        model = KNeighborsClassifier() # init model
        model.fit(x_data[:,i], y_true) # fit data
        results = permutation_importance(model, # permutation
                                         x_data[:,i], y_true, scoring='accuracy')
        importances[i] =  results.importances_mean # get importances


if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        obj = FeatureSelection(sys.argv[1]) # instantiate and pass main path
        obj.multi_folder() # get catalogue for multiple folders
    else:
        print('Please provide parent directory')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


