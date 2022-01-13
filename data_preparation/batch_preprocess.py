# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:43:22 2020

@author: panton01
"""

### ------------------- Imports ------------------- ###
import os, sys, json
from tqdm import tqdm
# User Defined
from helper.io_getfeatures import get_data, save_data
from data_preparation.preprocess import preprocess_data
from data_preparation.multich_data_prep import Lab2Mat
### ----------------------------------------------- ###


def batch_clean_filt(property_dict, num_channels = [0,1]):
    """
    batch_clean_filt(property_dict, num_channels = [0,1])

    Parameters
    ----------
    property_dict : Dictionary with essential paths
    num_channels : List, Channels to be filtered optional The default is [0,1].
    # Get channels by name

    Returns
    -------
    None.

    """
    
    # Get main path and load properties file
    jsonpath = os.path.join(property_dict['main_path'], 'organized.json') # name of dictionary where propeties are stored
    obj_props = Lab2Mat.load(jsonpath) # load dict
    
    # Get paths from dict
    main_path = property_dict['main_path'] # main path
    read_dir = property_dict['org_rawpath'] # name of read dir
    filt_dir = property_dict['filt_dir'] # name filt directory 
    
    # update and save dir as json
    obj_props.update({'filt_dir' : property_dict['filt_dir']}) # filter directory
    obj_props.update({'ch_list' : num_channels}) # channel list
    open(jsonpath, 'w').write(json.dumps(obj_props))
    
    # Create path if it doesn't exist
    save_dir = os.path.join(main_path, filt_dir)
    if os.path.isdir(save_dir) == 0:
        os.mkdir(save_dir)
    
    # Get file list 
    ver_path = os.path.join(main_path, read_dir)
    filelist = list(filter(lambda k: '.h5' in k, os.listdir(ver_path))) # get only files with predictions
    filelist = [os.path.splitext(x)[0] for x in filelist] # remove csv ending
    
    print('\n --->', len(filelist), 'files will be filtered.\n')
    for i in tqdm(range(0, len(filelist)), desc = 'Progress:', file=sys.stdout): # loop through experiments 
    
        # Get data and true labels
        data = get_data(main_path, filelist[i], ch_num = num_channels, inner_path={'data_path':read_dir}, load_y = False)
        
        # Clean and filter data
        data = preprocess_data(data,  clean = True, filt = True, verbose = 0)
        
        # Save data
        save_data(save_dir, filelist[i], data)
        
    print('Files in', main_path, 'directory have been cleaned and saved in:', '-/', filt_dir, '-')
    print('---------------------------------------------------------------------------\n')
        
        
if __name__ == '__main__':
    
    ### ------ USER INPUT ------ ###
    property_dict = {
    'main_path' : '', # path to parent dir
    'org_rawpath' : 'reorganized_data', # converted .h5 files
    'filt_dir' : 'filt_data', # filt directory 
    } 
        
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
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        