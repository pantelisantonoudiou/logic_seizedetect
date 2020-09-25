# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:15:00 2020

@author: panton01
"""

import sys
from error_check import ErrorCheck

property_dict = {
    'raw_data_path' : 'raw_data', # raw data directory
    'ch_struct' : ['vhpc', 'fc', 'emg'], # channel structure
    'file_ext' : '.adicht', # file extension
    'win' : 5, # window size in seconds
    'new_fs': 100, # new sampling rate
    'chunksize' : 2000, # number of rows to be read into memory
                 } 

class DataPrep():
    
     def __init__(self, property_dict):
         
        # get sub directories
        self.folders = [f.path for f in os.scandir(property_dictp) if f.is_dir()]
        
     def file_check(self):
        
        for f_path in self.folders:
            # if path exists
            if os.path.isdir(f_path) == 1:
                
            else:
                print('Path does not exist, please enter a valid path')
        
    
        # create instance
        obj = ErrorCheck(property_dict)
    
        # run analysis
        obj.mainfunc()


if __name__ == '__main__':
    
    # # get path from user
    # main_path = input('Enter data path:')
    
    if len(sys.argv)>1:
        main_path = sys.argv[1]
        
        # get sub directories
        folders = [f.path for f in os.scandir(main_path) if f.is_dir()]
        
        for f_path in folders:
            # if path exists
            if os.path.isdir(f_path) == 1:
                batch_clean_filt(f_path, num_channels = [0,1]) # filter and save files
            else:
                print('Path does not exist, please enter a valid path')
    else:
        print('Path was not entered')