# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:58 2020

@author: panton01
"""

import adi, os, sys, tables
from path_helper import get_dir
import numpy as np

property_dict = {
    'raw_data_path' : r'W:\Maguire Lab\Trina\2020\06- June\5142_5143_5160_5220\raw_data', # raw data path
    'ch_struct' : ['vhpc', 'fc', 'emg'], # channel structure
    'file_ext' : '.adicht', # file extension
    'win' : 5, # window size in seconds
                 } 

class ErrorCheck:
    
    # class constructor (data retrieval)
    def __init__(self, property_dict):
        """
        ErrorCheck(main_path)

        Parameters
        ----------
        property_dict : dict, with essential variables for file check

        """
        # Declare instance properties
        self.raw_data_path = property_dict['raw_data_path']
        self.ch_struct = property_dict['ch_struct'] # channel structure
        self.win = property_dict['win'] # channel structure
        self.file_ext = property_dict['file_ext'] # file extension
        self.chunksize = 2000 # number of rows to be read into memory
        
        # Get animal path
        animal_path = get_dir(self.raw_data_path,2)
        self.animal_ids = animal_path.split('_')

        # Get file list
        self.filelist = list(filter(lambda k: self.file_ext in k, os.listdir(self.raw_data_path)))
        
        # Get adi file obj to retrieve settings
        f = adi.read_file(os.path.join(self.raw_data_path, self.filelist[0]))
        
        # get sampling rate and downsample factor
        self.fs = round(f.channels[0].fs[0])
        self.down_factor = round(self.fs/self.down_factor)
        

def file_check(self):
    """

    Returns
    -------
    None.

    """
    
    # loop through labchart files (multilple animals per file)
    for i in range(len(self.filelist)):
        
        # get adi file obj
        f = adi.read_file(os.path.join(self.load_path, self.filelist[i])) 
        
        # get channel list
        ch_idx = np.linspace(1,f.n_channels,f.n_channels,dtype=int)
        ch_list = np.split(ch_idx, round(f.n_channels/len(self.ch_struct)))
        
        if len(ch_list) - len(self.animal_ids) != 0:
            print('Animal numbers do not match channel structure')
            
        for ii in range(len(ch_list)): # iterate through animals
            
            # get exp name
            filename = self.filelist[i].replace(self.file_ext, "") + '_' + self.animal_ids[ii]
     
            # downsample and save in chuncks
            self.test_files(f,filename,ch_list[ii])

    
    
    # save in chunks per animal
    def test_files(self,file_obj,filename,ch_list):
        """
        save_chunks(self,file_obj,filename,ch_list)

        Parameters
        ----------
        file_obj : ADI file object
        filename : String
        ch_list : List of numpy arrays 
            Containing channels for each animal.
            e.g. [1,2,3], [4,5,6]...

        Returns
        -------
        None.

        """
        
        ch_list = ch_list - 1 # convert channel to python format
        all_blocks = len(file_obj.channels[0].n_samples) # get all blocks
        
        for block in range(all_blocks):
            
            # get first channel (applies across animals channels)
            chobj = file_obj.channels[ch_list[0]] # get channel obj
            
            try: # skip corrupted blocks
                test = chobj.get_data(block+1,start_sample=0,stop_sample=1000)
            except:
                print(block, ' is corrupted')
                continue



def check_animal_ids(raw_data_path, ch_struct):
    
        # get animal directory
        animal_dir = get_dir(raw_data_path,1)
        
        # retreive animal ids
        animal_ids = animal_dir.split('_')
        
        
        self.filelist = list(filter(lambda k: self.file_ext in k, os.listdir(load_path)))
        
        # check if matches channel structure
        
        
        