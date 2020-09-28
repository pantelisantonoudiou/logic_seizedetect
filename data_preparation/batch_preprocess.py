# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:43:22 2020

@author: panton01
"""

### ------------------- Imports ------------------- ###
import os, sys
from tqdm import tqdm
# User Defined
parent_path = os.path.dirname(os.path.abspath(os.getcwd()))
if ( os.path.join(parent_path,'helper') in sys.path) == False:
    sys.path.extend([parent_path, os.path.join(parent_path,'helper')])
from io import get_data, save_data
from preprocess import preprocess_data
### ----------------------------------------------- ###

property_dict = {
    'main_path' : '', # parent dir
    'org_rawpath' : 'reorganized_data', # converted .h5 files
    'filt_dir' : 'filt_data', # filt directory 
    } 

def batch_clean_filt(property_dict, num_channels = [0,1]):
    
    main_path = property_dict['main_path'] # main path
    read_dir = property_dict['org_rawpath'] # name of read dir
    filt_dir = property_dict['filt_data'] # name filt directory 
    
    # create path if it doesn't exist
    save_dir = os.path.join(main_path, filt_dir)
    if os.path.isdir(save_dir) == 0:
        os.mkdir(save_dir)
    
    # get file list 
    ver_path = os.path.join(main_path, read_dir)
    filelist = list(filter(lambda k: '.h5' in k, os.listdir(ver_path))) # get only files with predictions
    filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
    
    print(len(filelist), 'files will be filtered.')
    for i in tqdm(range(0, len(filelist)), desc = 'Progress:', file=sys.stdout): # loop through experiments 
    
        # Get data and true labels
        data = get_data(main_path, filelist[i], ch_num = num_channels, inner_path={'data_path':read_dir}, load_y = False)
        
        # Clean and filter data
        data = preprocess_data(data,  clean = True, filt = True, verbose = 0)
        
        # Save data
        save_data(save_dir, filelist[i], data)
        
    print('Files in', main_path, 'directory have been cleaned and saved in:', '-', filt_dir, '-')
    print('---------------------------------------------------------------------------\n')
        
        
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
                property_dict['main_path'] = f_path # pass main path to properties dictionary
                batch_clean_filt(property_dict, num_channels = [0,1]) # filter and save files
            else:
                print('Path does not exist, please enter a valid path')
    else:
        print('Path was not entered')
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        