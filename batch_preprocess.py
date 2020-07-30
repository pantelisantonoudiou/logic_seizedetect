# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:43:22 2020

@author: panton01
"""

import os, sys
from tqdm import tqdm
from preprocess import preprocess_data
from build_feature_data import get_data, save_data

# path =  r'W:\Maguire Lab\Trina\2019\07-July\3514_3553_3639_3640'  # 3514_3553_3639_3640  3642_3641_3560_3514

def batch_clean_filt(main_path, num_channels = [0,1]):
    
    filt_dir = 'filt_data' # name filt directory 
    
    # create path if it doesn't exist
    save_dir = os.path.join(main_path, filt_dir)
    if os.path.isdir(save_dir) == 0:
        os.mkdir(save_dir)
    
    # get file list 
    ver_path = os.path.join(main_path, 'verified_predictions_pantelis')
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path))) # get only files with predictions
    filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
    
    print(len(filelist), 'files will be filtered.')
    print ('Processing ...')
    for i in tqdm(range(0, len(filelist))): # loop through experiments 
    
        # Get data and true labels
        data, y_true = get_data(main_path,filelist[i],ch_num = num_channels)
        
        # Clean and filter data
        data = preprocess_data(data,  clean = True, filt = True, verbose = 0)
        
        # Save data
        save_data(save_dir, filelist[i], data)
        
    print('Files in', main_path, 'directory have been cleaned and saved in', '-',filt_dir, '-')
        
        
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
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        