# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:22:20 2020

@author: Pante
"""

##    >>>>>>>>> USER INPUT <<<<<<<<          ##
property_dict = {
    'main_path' : '',       # parent path
    'data_dir' : 'raw_data', # raw data directory
    'org_rawpath' : 'reorganized_data', # converted .h5 files
    'rawpred_path': 'raw_predictions', # seizure predictions directory
    'filt_dir' : 'filt_data', # filt directory 
    'ch_struct' : ['vhpc', 'fc', 'emg'], # channel structure
    'file_ext' : '.adicht', # file extension
    'win' : 5, # window size in seconds
    'new_fs': 100, # new sampling rate
    'chunksize' : 2000, # number of rows to be read into memory
    'ch_list': [0,1]
                 } 
               ## ---<<<<<<<< ##              
               
### ------------------------ IMPORTS -------------------------------------- ###               
import os, sys, json
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# User Defined
parent_path = os.path.dirname(os.path.abspath(os.getcwd()))
if ( os.path.join(parent_path,'helper') in sys.path) == False:
    sys.path.extend([parent_path, os.path.join(parent_path,'helper')]) 
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
    def __init__(self, property_dict):
        """
        lab2mat(main_path)

        Parameters
        ----------
        property_dict : Dict, contains essential info for data load and processing

        """
        
        # Get main path and load properties file
        self.gen_path = property_dict['main_path']
        jsonpath = os.path.join(self.gen_path, 'organized.json') # name of dictionary where propeties are stored
        obj_props = Lab2Mat.load(jsonpath) # load dict
        
        # Pass input path
        self.org_rawpath = os.path.join(self.gen_path, property_dict['org_rawpath'])
               
        # Need to get from obj props
        self.filt_dir = property_dict['filt_dir'] # get filt dir
        self.ch_list = property_dict['ch_list'] # Get ch list from dict
        
        # Create raw pred path
        obj_props.update({'rawpred_path' : property_dict['rawpred_path']})
        self.rawpred_path = os.path.join(self.gen_path, property_dict['rawpred_path'])
        
        # get sampling rate
        self.fs = round(obj_props['fs'] / obj_props['down_factor'])
        self.win = obj_props['win']

        # Read method parameters into dataframe
        df = pd.read_csv('selected_method.csv')
        self.thresh = np.array(df.loc[0][df.columns.str.contains('Thresh')])
        self.weights = np.array(df.loc[0][df.columns.str.contains('Weight')])
        self.enabled = np.array(df.loc[0][df.columns.str.contains('Enabled')])
        
        # Get feature names
        self.feature_names = df.columns[df.columns.str.contains('Enabled')]
        self.feature_names = np.array([x.replace('Enabled_', '') for x in  self.feature_names])
        
        # write attributes to json file using a dict
        open(jsonpath, 'w').write(json.dumps(obj_props))


    def mainfunc(self, model_path):
        """
        mainfunc(input_path,model_path,ch_sel)
    
        Parameters
        ----------
        input_path : String, Path to raw data.

        model_path : String, Path to model.
        """
       
        # Create path prediction path
        if os.path.exists(self.rawpred_path) is False:
            os.mkdir(self.rawpred_path)
        
        # Get file list
        filelist = list(filter(lambda k: '.h5' in k, os.listdir(self.load_path)))
        
        # loop files (multilple channels per file)
        for i in tqdm(range(len(filelist)), desc = 'Progress', file=sys.stdout):
            
            # Get predictions (1D-array)
            data, bounds_pred = self.get_feature_pred(filelist.replace('.h5',''))
            
            # Convert prediction to binary vector and save as .csv
            self.save_idx(os.path.join(self.rawpred_path, filelist.replace('.h5','.csv')), data, bounds_pred)
            
               
    def get_feature_pred(self, file_id):
        """
        get_feature_pred(self, file_id)

        Parameters
        ----------
        file_id : Str, file name with no extension

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
        y_pred_array = (x_data > thresh)                            # get predictions for all conditions
        y_pred = y_pred_array * self.weights * self.enabled         # get predictions based on weights and selected features
        y_pred = np.sum(y_pred, axis=1) / np.sum(self.weights * self.enabled) # normalize to weights and selected features
        y_pred = y_pred > 0.5                                       # get popular vote
        bounds_pred = find_szr_idx(y_pred, np.array([0,1]))         # get predicted seizure index
        
        # If seizures are detected proceed to refine them
        if bounds_pred.shape[0] > 0:
            
            # Merge seizures close together
            bounds_pred = merge_close(bounds_pred, merge_margin = 5)
            
        return bounds_pred 

            
    def save_idx(file_path, data, bounds_pred):
        """
        Save user predictions to csv file as binary
    
        Parameters
        ----------
        file_path : Str, path to file save
        data : 3d Numpy Array (1D = segments, 2D = time, 3D = channel)
        bounds_pred : 2D Numpy Array (rows = seizures, cols = start and end points of detected seizures) 
        
        Returns
        -------
        None.
    
        """
        # pre allocate file with zeros
        ver_pred = np.zeros(data.shape[0])
    
        for i in range(bounds_pred.shape[0]): # assign index to 1
        
            if bounds_pred[i,0] > 0:
                # add 1 because csv starts from 1
                ver_pred[bounds_pred[i,0]+1:bounds_pred[i,1]+1] = 1
            
        # save file
        np.savetxt(file_path, ver_pred, delimiter=',',fmt='%i')

           
# Execute if module runs as main program
if __name__ == '__main__':

    if len(sys.argv) == 2:
    
        # update dict with raw path
        property_dict['main_path'] = sys.argv[1]
         
        # create instance
        obj = modelPredict(property_dict)
        
        # run analysis
        obj.mainfunc()
    
    else:
        print(' ---> Please provide parent directory.\n')
 

    
   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            