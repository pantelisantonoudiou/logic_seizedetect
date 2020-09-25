# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:39:58 2020

@author: panton01
"""

### -------------------- IMPORTS -------------------- ###
import os, sys
from tqdm import tqdm
import numpy as np
# User Defined
parent_path = os.path.dirname(os.path.abspath(os.getcwd()))
if ( os.path.join(parent_path,'helper') in sys.path) == False:
    sys.path.extend([parent_path, os.path.join(parent_path,'helper')])
import adi
from path_helper import get_dir, rem_array
#### ------------------------------------------------ ###

property_dict = {
    'raw_data_path' : r'W:\Maguire Lab\Trina\2020\06- June\5142_5143_5160_5220\raw_data', # raw data path
    'ch_struct' : ['vhpc', 'fc', 'emg'], # channel structure
    'file_ext' : '.adicht', # file extension
    'win' : 5, # window size in seconds
    'new_fs': 100, # new sampling rate
    'chunksize' : 2000, # number of rows to be read into memory
                 } 

class ErrorCheck:
    """
    ErrorCheck. Class for checking if labchart files can be parsed and read properly
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
        self.raw_data_path = property_dict['raw_data_path'] # raw path
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
        
        # Get sampling rate and downsample factor
        self.fs = round(f.channels[0].fs[0])
        self.down_factor = round(self.fs/property_dict['new_fs'])
        
    def increase_cntr(self):
        """
        """
        self.cntr += 1
           
    def mainfunc(self, full_check = False):
        """
        mainfunc(self)
        
        Parameters
        ----------
        full_check, bool = False, if True check that all files can be read

        Returns
        -------
        None.

        """
       
        print('------------------------- Initiating Error Check for ', self.raw_data_path, ' -------------------------\n')
       
        print('---> Step 1 : Testing file opening ... \n')
        
        # check that all blocks can be read or skipped succesfully
        success = self.file_check(self.test_files)
        
        if success is True:
            print('\n--- >', self.cntr-1, 'files were opened or skipped successfully.\n')
        
        
        if full_check is True:
            
            print('---> Step 2 : Testing file read ... \n')
            
            # check that files can be read in full
            success = self.file_check(self.test_full_read)
            if success is True:
                print('\n--- > All files were read successfully.\n')
        
        print('------------------------- Error Check Completed -------------------------\n')
                    
    def file_check(self, test_func):
        """
        file_check(self, test_func)
        
        Parameters
        ----------
        test_func, function reference for file testing
        
        Returns
        -------
        bool, True if operation succesfull 
    
        """
        self.cntr = 1 # init file counter
        
        # loop through labchart files (multilple animals per file)
        for i in  range(len(self.filelist)):
            
            # get adi file obj
            f = adi.read_file(os.path.join(self.raw_data_path, self.filelist[i])) 
            
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
            
            # print file being analyzed
            print(self.cntr,'-> Reading from block :', block, 'in File:', filename)
            
            # get first channel (applies across animals channels)
            chobj = file_obj.channels[ch_list[0]] # get channel obj
            length = chobj.n_samples[block] # get block length in samples
            
            try: # skip corrupted blocks
                chobj.get_data(block+1, start_sample=0, stop_sample=1000) # start
                chobj.get_data(block+1, start_sample=int(length/2), stop_sample=int(length/2)+1000) # middle
                chobj.get_data(block+1, start_sample=length-1000, stop_sample=length-1) # end
                self.increase_cntr() # increase object counter
            except:
                print('Block :', block, 'in File:', filename, 'is corrupted')
                continue

         
    # read full files
    def test_full_read(self, file_obj, filename, ch_list):
        """
        test_full_read(self, file_obj, filename, ch_list)

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
            
            print(self.cntr,'-> Reading block :', block, 'in File:', filename)
            
            # get first channel (applies across animals channels)
            chobj = file_obj.channels[ch_list[0]] # get channel obj
            
            try: # skip corrupted blocks
                chobj.get_data(block+1, start_sample=0, stop_sample=1000)
                self.increase_cntr() # increase object counter
            except:
                print('Block :', block, 'in File:', filename, 'is corrupted')
                continue
            
            # get channel length parameters and derive number of chuncks
            length = chobj.n_samples[block] # get block length in samples
            win_samp = self.win * self.fs # get window size in samples
            mat_shape = [0,0] # init mat shape
            mat_shape[0] = int(np.floor(length/win_samp)) # get number of rows
            mat_shape[1] = round(win_samp / self.down_factor) # get number of columns
            idx = rem_array(0, mat_shape[0], self.chunksize) # get index
            
            ## Iterate over channel length ##
            for i in tqdm(range(len(idx)-1), desc = 'Progress', file=sys.stdout): # loop though index 

                # read channel chunk
                self.get_filechunks(file_obj.channels[ch_list[0]], block+1, mat_shape[1], idx[i:i+2])
    
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
        
# Execute if module runs as main program
if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        
        # update dict with raw path
        property_dict['main_path'] = sys.argv[1]
     
        # create instance
        obj = ErrorCheck(property_dict)
    
        # run analysis
        obj.mainfunc()

    else:
        print(' ---> Please provide parent directory.\n')
 
    

   














        
        
        