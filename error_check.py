# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:58 2020

@author: panton01
"""

### -------------------- IMPORTS -------------------- ###
import adi, os, sys
from tqdm import tqdm
import numpy as np
from path_helper import get_dir, rem_array
#### ------------------------------------------------ ###

property_dict = {
    'raw_data_path' : r'W:\Maguire Lab\Trina\2020\06- June\5142_5143_5160_5220\raw_data', # raw data path
    'ch_struct' : ['vhpc', 'fc', 'emg'], # channel structure
    'file_ext' : '.adicht', # file extension
    'win' : 5, # window size in seconds
    'chunksize' : 2000 # number of rows to be read into memory
                 } 

class ErrorCheck:
    """
    Class for checking if labchart files can be parsed and read properly
    One labchart file may contain recordings from multiple animals
    Each animal may have multiple channels (eg. 3 channels per animal)
    Hence one example file can have 12 channels from 4 animals
    """
    
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
        self.chunksize = property_dict['chunksize'] # number of rows to be read into memory
        
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
        
        
    def main_func(self):
        """
        main_func(self)

        Returns
        -------
        None.

        """
       
        print('-------------- 1/2 -> Testing file read -------------------')
        
         # check that all blocks can be read or skipped succesfully
        success = self.file_check(self.test_files)
        
        if success is True:
            print('Files and blocks opened/skipped successfully.\n')
            
        print('--------------------------------------------------\n')
        
        
        print('-------------- 2/2 ->Testing full file read -------------------')
        
        # check that files can be read in full
        success = self.file_check(self.test_full_read)
        if success is True:
            print('Files read successfully.\n')
            
        print('-------------Error Check complete---------------------\n')
            

    def file_check(self, test_func):
        """
        
        Parameters
        ----------
        test_func, function reference for file testing
        
        Returns
        -------
        bool, true if operation succesfull 
    
        """
        
        # loop through labchart files (multilple animals per file)
        for i in  tqdm(range(len(self.filelist)), desc = 'File', file=sys.stdout):
            
            # get adi file obj
            f = adi.read_file(os.path.join(self.load_path, self.filelist[i])) 
            
            # get channel list
            ch_idx = np.linspace(1,f.n_channels,f.n_channels,dtype=int)
            ch_list = np.split(ch_idx, round(f.n_channels/len(self.ch_struct)))
            
            if len(ch_list) - len(self.animal_ids) != 0:
                print('Error! Animal numbers do not match channel structure')
                return False
                
            for ii in range(len(ch_list)): # iterate through animals
                
                # get exp name
                filename = self.filelist[i].replace(self.file_ext, "") + '_' + self.animal_ids[ii]
         
                # downsample and save in chuncks
                test_func(f,filename,ch_list[ii])
                
        return True

    
    # read the start of files/blocks
    def test_files(self, file_obj, filename, ch_list):
        """
        test_files(self, file_obj, filename, ch_list)

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
                chobj.get_data(block+1,start_sample=0,stop_sample=1000)
            except:
                print(block, 'in', filename, 'is corrupted')
                continue

         
    # read full files
    def test_full_read(self ,file_obj, filename, ch_list):
        """
        test_full_read(self ,file_obj, filename, ch_list)

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
                chobj.get_data(block+1, start_sample=0, stop_sample=1000)
            except:
                print(block, 'in', filename, 'is corrupted')
                continue
            
            # get channel length parameters and derive number of chuncks
            length = chobj.n_samples[block] # get block length in samples
            win_samp = self.win * self.fs # get window size in samples
            mat_shape = np.floor(length/win_samp) # get number of rows
            idx = rem_array(0, mat_shape, self.chunksize) # get index
            
            ## Iterate over channel length ##
            for i in tqdm(range(len(idx)-1), desc = 'Experiment', file=sys.stdout): # loop though index 

                for ii in range(len(ch_list)): ## Iterate over all animal channels ##
                    # get channel obj
                    chobj = file_obj.channels[ch_list[ii]] 
                    
                    # get data per channel
                    self.get_filechunks(chobj,block+1,mat_shape[1],idx[i:i+2])
    
    
    # read labchart segment
    def get_filechunks(self, chobj, block, cols, idx):
        """
        get_filechunks(self, chobj, block, cols, idx)

        Parameters
        ----------
        chobj : ADI Channel object
 
        block : Int, Block number of labchart file.
        cols : Int, number of columns.
        idx : ndarray, start and stop index in window blocks.

        """
        
        # copy index and change to samples
        index = idx.copy()
        if index[0] == 0:    
            index[1] *= (self.fs * self.win)
            index[0] = 1
        else:          
            index = index * (self.fs * self.win)
            index[1] -=1

        # retrieve data
        chobj.get_data(block,start_sample = index[0], stop_sample = index[1])
        
        
        
        
        