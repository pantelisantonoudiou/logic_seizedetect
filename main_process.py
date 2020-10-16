# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:15:00 2020

@author: panton01
"""

### ------------------- Imports ------------------- ###
import os, json
from data_preparation.error_check import ErrorCheck
from data_preparation.multich_data_prep import Lab2Mat
from data_preparation.batch_preprocess import batch_clean_filt
from data_preparation.get_predictions import modelPredict
### ----------------------------------------------- ###

# property_dict = {
#     'data_dir' : 'raw_data', # raw data directory
#     'org_rawpath' : 'reorganized_data', # converted .h5 files
#     'rawpred_path': 'raw_predictions', # seizure predictions directory
#     'main_path' : '',       # parent path
#     'filt_dir' : 'filt_data', # filt directory 
#     'ch_struct' : ['vhpc', 'fc', 'emg'], # channel structure
#     'file_ext' : '.adicht', # file extension
#     'win' : 5, # window size in seconds
#     'new_fs': 100, # new sampling rate
#     'chunksize' : 2000, # number of rows to be read into memory
#     'ch_list': [0,1],
#     'properties_file':'organized.json',
#                  } 

class DataPrep():
    """
    """
    
    def __init__(self, main_path, property_dict):
        
        # pass main path to object
        self.main_path = main_path
        
        # get sub directories
        self.folders = [f.path for f in os.scandir(self.main_path) if f.is_dir()]
        
        # pass properties dictionary to object
        self.properties = property_dict
        
    def file_check(self):
        """

        Returns
        -------
        Bool, True if all files checked are correct

        """
        
        print('---------------------------------------------------------------------------\n')
        print('------------------------- Initiating Error Check  -------------------------\n')
        print('--->', len(self.folders), 'folders will be checked.\n')
        
        success_list = [] # init lsit of bools for success of each folder file check
        for f_path in self.folders: # iterate over folders
            
            if os.path.isdir(f_path) == 1: # if path exists
                
                # pass main path to dict               
                self.properties['main_path'] = f_path
    
                # check files
                obj = ErrorCheck(self.properties) # intialize object
                success = obj.mainfunc() # run file check
                success_list.append(success) # append to list
   
        print('-------------------------- Error Check Completed --------------------------\n')
        print('***************************** Succesful =', all(success_list), '*****************************\n')
        return all(success_list)

def main_func(main_path, property_dict):
    """

    Parameters
    ----------
    main_path : Str, Path to parent directory containing animal folders
    property_dict: Dict, Configuration settings (Check Documentation for info)

    Returns
    -------
    bool, False if operation fails.

    """
    
    # File Check
    file_check_success = False
    
    if os.path.isdir(main_path): 
        
        # instantiate object
        obj = DataPrep(main_path, property_dict) 
        try:
            # perform file check
            file_check_success = obj.file_check() 
            
        except Exception as e:
            
            print('---> File check Failed! Operation Aborted.\n')
            print(e + '\n')

    else:
        print('\n************ The input', '"'+ main_path +'"' ,'is not a valid path. Please try again ************\n')
        return False 
    
    if file_check_success is True:   
        
        answer = '' # init empty answer
        while answer != 'y' and answer != 'n': # repeat until get yes or no
        
            # Verify whether to proceed
            answer = input('-> File Check Completed. Dou want to proceed (y/n)? \n') 
        
            if answer == 'y':
                for f_path in obj.folders: # iterate over folders      
                    if os.path.isdir(f_path) == 1: # if path exists
                        
                        # -1 Convert Labchart to .h5 objects
                        property_dict['main_path'] = f_path # update dict with main path
                        file_obj = Lab2Mat(property_dict) # instantiate object    
                        file_obj.mainfunc() # Convert data   
    
                        # -2 Filter and preprocess data
                        batch_clean_filt(property_dict,  num_channels = property_dict['ch_list'])
                        
                        # -3 Get Method/Model Predictions
                        model_obj = modelPredict(property_dict)
                        model_obj.mainfunc() # Get predictions
                        
                print('\n************************* All Steps Completed *************************n')
                
            elif answer == 'n':
                print('---> No Further Action Will Be Performed.\n')

if __name__ == '__main__':
    
    # Load config file 
    try:
        property_dict = open('config.json', 'r').read()
        property_dict = json.loads(property_dict)
    except Exception as err:
        raise FileNotFoundError(f"Unable to read the config file.\n{err}")
    
    # Get path from user
    main_path = input('Enter data path:')
    
    # Run main script for file check, conversion to .h5, fitlering and predictions 
    main_func(main_path, property_dict)

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                