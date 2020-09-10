# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:11:27 2020
@author: panton01
"""

### ----------------------- USER INPUT --------------------------------- ###

# --->>>>>>>>> 
# Add path to raw data folder in following format -> r'PATH'
input_path = r'C:\Users\panton01\Desktop\Trina-seizures\3642_3641_3560_3514\raw_data'
                # ---<<<<<<<< #

### ------------------------------------------------------------------------###

### ------------------------ IMPORTS -------------------------------------- ###
import adi,os,sys,tables,json
from tqdm import tqdm
import numpy as np
from scipy import signal
from math import floor
from string import ascii_lowercase
from path_helper import get_dir,sep_dir,rem_array
### ------------------------------------------------------------------------###


## print animal to be analyzed ##

# CLASS FOR CONERTING LABCHART TO H5 FILES
class lab2mat:
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
    def __init__(self, load_path):
        """
        lab2mat(main_path)

        Parameters
        ----------
        load_path : STRING
            PATH.

        """
        # Declare instance properties
        self.ch_struct = ['vhpc', 'fc', 'emg'] # channel structure
        self.win = 5 # seconds
        self.fs = 0 # sampling rate
        self.down_factor = 100 # downsample factor
        self.animal_ids = [] # animal IDs in folder
        self.load_path ='' # full load path
        self.save_path = '' # full save path
        self.gen_path = '' # general path (animal level)
        self.raw_path = '' # folder containing labchart files
        self.org_rawpath = 'reorganized_data' # reorganized data folder
        self.filelist = [] # file list
        self.file_ext = '.adicht' # file extension
        self.chunksize = 2000 # number of rows to be read into memory
        
        # Get general and inner paths
        self.gen_path, self.raw_path = sep_dir(load_path,1)
        
        # Get animal path
        temp = get_dir(load_path,2)
        self.animal_ids = temp.split('_')

        # Make paths
        self.load_path = os.path.join(self.gen_path, self.raw_path)
        self.save_path = os.path.join(self.gen_path, self.org_rawpath)
        self.filelist = list(filter(lambda k: self.file_ext in k, os.listdir(load_path)))
        
        # Get adi file obj to retrieve settings
        f = adi.read_file(os.path.join(load_path, self.filelist[0]))
        
        # get sampling rate and downsample factor
        self.fs = round(f.channels[0].fs[0])
        self.down_factor = round(self.fs/self.down_factor)
        
  
    # main methods (iterate over files)
    def mainfunc(self):
        """

        Returns
        -------
        None.

        """
        
        # make path
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        
        # init progress bar
        total = len(self.filelist)*len(self.animal_ids)       
        
        cntr = 1 # init counter
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
                self.save_chunks(f,filename,ch_list[ii])
                
                print(cntr, 'of',total ,'Experiments Saved')
                cntr += 1 # update counter
    
    
    # save in chunks per animal
    def save_chunks(self,file_obj,filename,ch_list):
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
            
            length = chobj.n_samples[block] # get block length in samples
            win_samp = self.win * self.fs # get window size in samples
            mat_shape = [0,0]
            mat_shape[0] = floor(length/win_samp) # get number of rows
            mat_shape[1] = round(win_samp / self.down_factor) # get number of columns
            idx = rem_array(0, mat_shape[0], self.chunksize) # get index
            
            
            ### SAVING PARAMETERS ##
            file_id  = filename + ascii_lowercase[block] + '.h5' # add extension
            full_path = os.path.join(self.save_path, file_id) # get full save path
            fsave = tables.open_file(full_path, mode='w') # open tables object
            atom = tables.Float64Atom() # declare data type     
            ds = fsave.create_earray(fsave.root, 'data',atom, # create data store 
                                        [0, mat_shape[1],len(ch_list)], 
                                        chunkshape = [self.chunksize,mat_shape[1],len(ch_list)])
            
            ## Iterate over channel length ##
            for i in tqdm(range(len(idx)-1), desc = 'Experiment', file=sys.stdout): # loop though index 
            
                # preallocate data
                data = np.zeros([idx[i+1] - idx[i], mat_shape[1], len(ch_list)])
                
                for ii in range(len(ch_list)): ## Iterate over all animal channels ##
                    # get channel obj
                    chobj = file_obj.channels[ch_list[ii]] 
                    
                    # get data per channel
                    data[:,:,ii] = self.get_filechunks(chobj,block+1,mat_shape[1],idx[i:i+2])
    
                # append data to datastore
                ds.append(data)
            
            # print('Total length = ',length)
            fsave.close() # close table object
        
    
    # segment labchart file to numpy array
    def get_filechunks(self,chobj,block,cols,idx):
        """
         get_filechunks(self,chobj,block,cols,idx)

        Parameters
        ----------
        chobj : ADI CHANNEL OBJECT
            DESCRIPTION.
        block : INT
            Block number of labchart file.
        cols : INT
            number of columns.
        idx : TWO ELEMENT NP VECTOR- INT
            start and stop index in window blocks.

        Returns
        -------
        data : numpy array
            DESCRIPTION.

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
   # create instance
   obj = lab2mat(input_path)
   
   # run analysis
   obj.mainfunc()
   
   # save attributes as dictionary        
   obj.save(os.path.join(obj.gen_path, 'organized.json'))
      



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        