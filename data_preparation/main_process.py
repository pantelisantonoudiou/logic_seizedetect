# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:15:00 2020

@author: panton01
"""

import os, sys
from error_check import ErrorCheck
from multich_dataPrep import lab2mat

property_dict = {
    'data_dir' : 'raw_data', # raw data directory
    'org_rawpath' : 'reorganized_data', # converted .h5 files
    'main_path' : '', # parent path
    'raw_data_path' : '', # raw data path
    'ch_struct' : ['vhpc', 'fc', 'emg'], # channel structure
    'file_ext' : '.adicht', # file extension
    'win' : 5, # window size in seconds
    'new_fs': 100, # new sampling rate
    'chunksize' : 2000, # number of rows to be read into memory
                 } 

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
                
        return all(success_list)
                
        print('-------------------------- Error Check Completed --------------------------\n')
        print('---------------------------------------------------------------------------\n')

def main_func(main_path):
    
    # File Check
    file_check_success = False
    if os.path.isdir(main_path): 
        obj = DataPrep(main_path, property_dict) # instantiate object
        try:
            file_check_success = obj.file_check() # perform file check
            print('***** file_check_success =', file_check_success)
        except:
            print('---> File check Failed! Operation Aborted.')
            print(sys.exc_info()[0])

    else:
        print('\n****** The input <', main_path ,'> is not a path. Please try again.******\n')
        return False 
    
    if file_check_success is True:
        # Verify whether to proceed
        answer = input('-> File Check Completed. Dou want to proceed? (y/n)\n')    
        
        if answer == 'y':
            for f_path in obj.folders: # iterate over folders      
                if os.path.isdir(f_path) == 1: # if path exists
                    
                    # Convert Labchart to .h5 objects
                    property_dict['main_path'] = f_path # update dict with main path
                    file_obj = lab2mat(property_dict) # instantiate object    
                    file_obj.mainfunc() # run analysis   
                    file_obj.save(os.path.join(property_dict['main_path'], 'organized.json')) # save attributes as dictionary  
                    
                    # Filter and preprocess data

if __name__ == '__main__':
    
    # Get path from user
    main_path = input('Enter data path:')
    
    # Run main script for file check, conversion to .h5, fitlering and predictions 
    main_func(main_path)

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                