# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:11:27 2020
@author: panton01
"""

### ----------------------- USER INPUT --------------------------------- ###
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
### ------------------------------------------------------------------------###

### ------------------------ IMPORTS -------------------------------------- ###
import os, sys, tables, json
from tqdm import tqdm
import numpy as np
from scipy import signal
from math import floor
from string import ascii_lowercase
# User Defined
parent_path = os.path.dirname(os.path.abspath(os.getcwd()))
if ( os.path.join(parent_path,'helper') in sys.path) == False:
    sys.path.extend([parent_path, os.path.join(parent_path,'helper')])
import adi
from path_helper import get_dir, rem_array
### ------------------------------------------------------------------------###


# CLASS FOR CONVERTING LABCHART TO H5 FILES
class Lab2Mat:
    """
    Class for convering labchart files to H5 format
    One labchart file may contain recordings from multiple animals
    Each animal may have multiple channels
    For example one file-> 1-12 channels and 4 animals
    One H5 file will contain all channels for 1 animal (eg. 1-3)
    
    -> __init__: constructor to get instance attributes
    -> mainfunc: function that iterates over labchart files
    -> save_chunks: save all channels in one animal in chuncks
    -> get_filechunks: get file chunk
    -> save: save object properties as dict (json format)
    -> load: load object properties from dict (stored in json format)
    """
   
    # class constructor (data retrieval)
    def __init__(self, property_dict):
        """
        lab2mat(main_path)

        Parameters
        ----------
        property_dict : Dict contianing essential parameters for conversion

        """
        # Declare instance properties
        self.ch_struct = property_dict['ch_struct'] # channel structure
        self.win = property_dict['win'] # seconds
        self.animal_ids = [] # animal IDs in folder
        self.load_path ='' # full load path
        self.save_path = '' # full save path
        self.org_rawpath = property_dict['org_rawpath'] # reorganized data folder
        self.file_ext = '.adicht' # file extension
        self.chunksize = property_dict['chunksize'] # number of rows to be read into memory
        
        # Get animal path
        self.animal_ids = get_dir(property_dict['main_path'],1).split('_')
        
        # Get raw data path
        self.load_path = os.path.join(property_dict['main_path'], property_dict['data_dir']) # raw path

        # Make paths
        self.save_path = os.path.join(property_dict['main_path'], self.org_rawpath)
        self.filelist = list(filter(lambda k: self.file_ext in k, os.listdir(self.load_path)))
        
        # Get adi file obj to retrieve settings
        f = adi.read_file(os.path.join(self.load_path, self.filelist[0]))
        
        # get sampling rate and downsample factor
        self.fs = round(f.channels[0].fs[0]) # sampling rate
        self.new_fs = property_dict['new_fs'] # new sampling rate
        self.down_factor = round(self.fs/self.new_fs)

    def increase_cntr(self):
        """
        """
        self.cntr += 1    
  
    # main methods (iterate over files)
    def mainfunc(self):
        """

        Returns
        -------
        Bool, False if channel list does not match channels structure

        """
        print('---------------------------------------------------------------------------\n')
        print('---> Initiating File Conversion for', self.load_path+'.', '\n')
        
        # make path
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        
        self.cntr = 1 # init counter
        
        # loop through labchart files (multilple animals per file)
        for i in range(len(self.filelist)):
            
            # get adi file obj
            f = adi.read_file(os.path.join(self.load_path, self.filelist[i])) 
            
            # get channel list
            ch_idx = np.linspace(1,f.n_channels,f.n_channels,dtype=int)
            ch_list = np.split(ch_idx, round(f.n_channels/len(self.ch_struct)))

            if len(ch_list) - len(self.animal_ids) != 0:
                print('******** Animal numbers do not match channel structure ********\n')
                return False
                
            for ii in range(len(ch_list)): # iterate through animals
                
                # get exp name
                filename = self.filelist[i].replace(self.file_ext, "") + '_' + self.animal_ids[ii]
         
                # downsample and save in chuncks
                # self.save_chunks(f,filename,ch_list[ii])

        print('\n--->  File Conversion Completed.', self.cntr-1, '\n Files Were Saved To:', self.save_path+'.', '\n')
        print('---------------------------------------------------------------------------\n')
        return True
    
    # save in chunks per animal
    def save_chunks(self, file_obj, filename, ch_list):
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
            
            # print file being analyzed
            print(self.cntr,'-> Converting block :', block, 'in File:', filename)
            self.increase_cntr() # increase object counter
            
            # get first channel (applies across animals channels)
            chobj = file_obj.channels[ch_list[0]] # get channel obj
            
            try: # skip corrupted blocks
                chobj.get_data(block+1,start_sample=0,stop_sample=1000)
            except:
                print('Block :', block, 'in File:', filename, 'is corrupted.')
                continue
            
            ### CHANNEL PARAMETERS ###
            length = chobj.n_samples[block] # get block length in samples
            win_samp = self.win * self.fs # get window size in samples
            mat_shape = [0,0] # init mat shape
            mat_shape[0] = floor(length/win_samp) # get number of rows
            mat_shape[1] = round(win_samp / self.down_factor) # get number of columns
            idx = rem_array(0, mat_shape[0], self.chunksize) # get index
            
            ### SAVING PARAMETERS ###
            file_id  = filename + ascii_lowercase[block] + '.h5' # add extension
            full_path = os.path.join(self.save_path, file_id) # get full save path
            fsave = tables.open_file(full_path, mode='w') # open tables object
            atom = tables.Float64Atom() # declare data type     
            ds = fsave.create_earray(fsave.root, 'data',atom, # create data store 
                                        [0, mat_shape[1],len(ch_list)], 
                                        chunkshape = [self.chunksize,mat_shape[1],len(ch_list)])
            
            ## Iterate over channel length ##
            for i in tqdm(range(len(idx)-1), desc = 'Progress', file=sys.stdout): # loop though index 
            
                # preallocate data
                data = np.zeros([idx[i+1] - idx[i], mat_shape[1], len(ch_list)])
                
                for ii in range(len(ch_list)): ## Iterate over all animal channels ##
                    # get channel obj
                    chobj = file_obj.channels[ch_list[ii]] 
                    
                    # get data per channel
                    data[:,:,ii] = self.get_filechunks(chobj,block+1,mat_shape[1],idx[i:i+2])
    
                # append data to datastore
                ds.append(data)
            
            # close table object
            fsave.close() 
        
    
    # segment labchart file to numpy array
    def get_filechunks(self,chobj,block,cols,idx):
        """
         get_filechunks(self,chobj,block,cols,idx)

        Parameters
        ----------
        chobj : ADI labchart chanel object
 
        block : Int, Block number of labchart file.
        cols : Int, number of columns.
        idx : TWO ELEMENT NP VECTOR- INT
            start and stop index in window blocks.

        Returns
        -------
        data : numpy array

        """
        
        # copy index and change to samples
        index = idx.copy()
        if index[0] == 0:    
            index[1] *= (self.fs * self.win)
            index[0] = 1
        else:          
            index = index * (self.fs * self.win)
            index[1] -=1

        # print(index)
        # retrieve data
        data = chobj.get_data(block,start_sample = index[0], stop_sample = index[1])
        
        # decimate data
        data = signal.decimate(data, self.down_factor)
        
        # get matrix shape
        mat_shape = [round(len(data)/cols), cols]
        
        # reshape data to matrix
        data = np.reshape(data, mat_shape)

        return data
        

   # save object attributes to path
    def save(self, path):
        to_export = self.__dict__       
        open(path, 'w').write(json.dumps(to_export))
        return True
    
    # load object attributes to memory
    @classmethod
    def load(self, path):
        openpath = open(path, 'r').read()
        attrb = json.loads(openpath)
        return attrb

# Execute if module runs as main program
if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        
        # update dict with raw path
        property_dict['main_path'] = sys.argv[1]
     
        # create instance
        obj = Lab2Mat(property_dict)
    
        # run analysis
        obj.mainfunc()
        
        # save attributes as dictionary        
        obj.save(os.path.join(property_dict['main_path'], 'organized.json'))

    else:
        print(' ---> Please provide parent directory.\n')


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        