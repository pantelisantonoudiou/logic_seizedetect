# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:15:00 2020

@author: panton01
"""

import os, sys
from error_check import ErrorCheck

property_dict = {
    'data_dir' : 'raw_data', # raw data directory
    'raw_data_path' : '', # full path to raw data
    'ch_struct' : ['vhpc', 'fc', 'emg'], # channel structure
    'file_ext' : '.adicht', # file extension
    'win' : 5, # window size in seconds
    'new_fs': 100, # new sampling rate
    'chunksize' : 2000, # number of rows to be read into memory
                 } 

class DataPrep():
    
     def __init__(self, main_path, property_dict):
        
        # pass main path to object
        self.main_path = main_path
        
        # get sub directories
        self.folders = [f.path for f in os.scandir(self.main_path) if f.is_dir()]
        
        # pass properties dictionary to object
        self.properties = property_dict
        
     def file_check(self):
        
        for f_path in self.folders: # iterate over folders
            
            if os.path.isdir(f_path) == 1: # if path exists
                
                # get path               
                self.properties['raw_data_path'] = os.path.join(self.main_path, self.properties['data_dir'])
            
                # check files
                obj = ErrorCheck(self.properties) # intialize object
                obj.mainfunc() # run file check


if __name__ == '__main__':
    
    # # get path from user
    main_path = input('Enter data path:')
    breakpoint()
    if os.path.isdir(main_path):
        
        obj = DataPrep(main_path, property_dict) # instantiate object
        
        try:
            obj.file_check() # perform file check
        except:
            print('---> File check Failed! Operation Aborted.')
    else:
        print('The input', main_path ,' was not a path. Please try again')
            
            
    #     # get sub directories
    #     folders = [f.path for f in os.scandir(main_path) if f.is_dir()]
        
    #     for f_path in folders:
    #         # if path exists
    #         if os.path.isdir(f_path) == 1:
    #             batch_clean_filt(f_path, num_channels = [0,1]) # filter and save files
    #         else:
    #             print('Path does not exist, please enter a valid path')
    # else:
    #     print('Path was not entered')