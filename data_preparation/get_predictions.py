# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:22:20 2020

@author: Pante
"""

##    >>>>>>>>> USER INPUT <<<<<<<<          ##
# # Add path to raw data folder in following format -> r'PATH'
# input_path = r'C:\Users\Pante\Desktop\test1\raw_data'

# Add path to model
# model_path = r'models\cnn_train_ratio_1_ch_1and2_train03.h5'

               ## ---<<<<<<<< ##              
               
### ------------------------ IMPORTS -------------------------------------- ###               
import os, sys, tables
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# User Defined
parent_path = os.path.dirname(os.path.abspath(os.getcwd()))
if ( os.path.join(parent_path,'helper') in sys.path) == False:
    sys.path.extend([parent_path, os.path.join(parent_path,'helper')])
from path_helper import sep_dir    
from array_helper import find_szr_idx, merge_close
from io_getfeatures import get_data, get_features_allch
from multich_data_prep import Lab2Mat
import features
### ------------------------------------------------------------------------###

              
class modelPredict:
    """
    Class for batch seizure prediction
    """
    
    # class constructor (data retrieval) ### GET CORRECT PARAMETERS
    def __init__(self, input_path):
        """
        lab2mat(main_path)

        Parameters
        ----------
        input_path : STRING
            Raw data path.

        """
        # pass input path
        self.input_path = input_path

        # Get general and inner paths
        self.gen_path, innerpath = sep_dir(input_path,1)
        
        # load object properties as dict
        jsonfile = 'organized.json'
        obj_props = Lab2Mat.load(os.path.join(self.gen_path, jsonfile))
        self.org_rawpath = obj_props['org_rawpath']
        
        # create raw pred path
        rawpred_path = 'model_predictions'
        obj_props.update({'rawpred_path' : rawpred_path})
        self.rawpred_path = os.path.join(self.gen_path, rawpred_path)
        
        # get sampling rate
        self.fs = round(obj_props['fs'] / obj_props['down_factor'])
        self.win = obj_props['win']
        
        self.surround_time = 60 # time around seizures in seconds
        self.szr_pwr_thresh = 200 # szr power thresh         
        self.szr_segments =  np.array([2,2],dtype=int)  # segments before and after seizure segment
        
        self.ch_list ;## get ch list from dict
        self.load_path # reorganized path
        
        # Read method parameters into dataframe
        df = pd.read_csv('selected_method.csv')
        self.thresh = np.array(df.loc[0][df.columns.str.contains('Thresh')])
        self.weights = np.array(df.loc[0][df.columns.str.contains('Weight')])
        self.enabled = np.array(df.loc[0][df.columns.str.contains('Enabled')])
        
        # Get feature names
        self.feature_names = df.columns[df.columns.str.contains('Enabled')] # get
        self.feature_names = np.array([x.replace('Enabled_', '') for x in  self.feature_names])


    def mainfunc(self, model_path):
        """
        mainfunc(input_path,model_path,ch_sel)
    
        Parameters
        ----------
        input_path : String, Path to raw data.

        model_path : String, Path to model.
        """
       
        # make path
        if os.path.exists( self.rawpred_path) is False:
            os.mkdir( self.rawpred_path)
        
        # get file list
        filelist = list(filter(lambda k: '.h5' in k, os.listdir(self.load_path)))
        
        # loop files (multilple channels per file)
        for i in tqdm(range(len(filelist))):
            
            # get predictions (1D-array)
            bin_pred = self.get_feature_pred(file_id)
   
            # save predictions as .csv
            file_id = filelist[i].replace('.h5', '.csv')
            np.savetxt(os.path.join(self.rawpred_path,file_id), ref_pred, delimiter=',',fmt='%i')
            
            

    ### add method    
    def get_feature_pred(self, file_id):
        """
        get_feature_pred(self, file_id)

        Parameters
        ----------
        file_id : Str

        Returns
        -------
        data : 3d Numpy Array (1D = segments, 2D = time, 3D = channel)
        bounds_pred : 2D Numpy Array (rows = seizures, cols = start and end points of detected seizures)

        """
        
        # Define parameter list
        param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
                      features.get_envelope_max_diff,) # single channel features
        cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,) # cross channel features
        
        # Get data and true labels
        data = get_data(self.gen_path, file_id, ch_num = self.ch_list, inner_path={'data_path':'filt_data'}, load_y = False)
        
        # Extract features and normalize
        x_data, labels = get_features_allch(data,param_list, cross_ch_param_list) # Get features and labels
        x_data = StandardScaler().fit_transform(x_data) # Normalize data
        
        # Get predictions
        thresh = (np.mean(x_data) + self.thresh * np.std(x_data))   # get threshold vector
        y_pred_array = (x_data > thresh) # get predictions for all conditions
        y_pred = y_pred_array * self.weights * self.enabled    # get predictions based on weights and selected features
        y_pred = np.sum(y_pred, axis=1) / np.sum(self.weights * self.enabled) # normalize to weights and selected features
        y_pred = y_pred > 0.5                                       # get popular vote
        bounds_pred = find_szr_idx(y_pred, np.array([0,1]))         # get predicted seizure index
        
        # If seizures are detected proceed to refine them
        if bounds_pred.shape[0] > 0:
            
            # Merge seizures close together
            bounds_pred = merge_close(bounds_pred, merge_margin = 5)
            
            # Remove seizures where a feature (line length or power) is not higher than preceeding region
            idx = np.where(np.char.find(self.feature_names,'line_length_0')==0)[0][0]
            bounds_pred = self.refine_based_on_surround(x_data[:,idx], bounds_pred)    
        
        return bounds_pred 

        
# # Execute if module runs as main program
# if __name__ == '__main__':
    
#     # init object
#     obj = modelPredict(input_path)
    
#     # get predictions in binary format and store in csv
#     obj.mainfunc(model_path)    
    
    
    
   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            