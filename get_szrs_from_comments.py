# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:00:13 2020

@author: panton01
"""


### ----------------------- USER INPUT --------------------------------- ###

# --->>>>>>>>> 
# Add path to raw data folder in following format -> r'PATH'
input_path = r'W:\Maguire Lab\Trina\2020\06- June\5142_5143_5160_5220\raw_data'
                # ---<<<<<<<< #

### ------------------------------------------------------------------------###

### ------------------------ IMPORTS -------------------------------------- ###
import adi, os, sys
from pprint import pprint
import numpy as np
import pandas as pd
from multich_dataPrep import lab2mat
from string import ascii_lowercase
from path_helper import sep_dir
### ------------------------------------------------------------------------###


# CLASS FOR RETRIEVING SEIZURE COMMENTS FROM LABCHART
class SzrsFromLab:
    """
    Class for retrieving seizure comments from lachart
    One labchart file may contain recordings from multiple animals
    Each animal may encompass multiple channels
    For example one file-> 1-12 channels and 4 animals
    
    -> __init__: constructor to get instance attributes
    -> mainfunc: function that iterates over labchart files
    -> extract_comments: get comments for all blocks of each animal
    -> filter_coms: Static, filter comments based on channel ID (eg. 1,10)
    """
   
    # class constructor (load_path retrieval)
    def __init__(self, input_path):
        """
        SzrsFromLab(main_path)

        Parameters
        ----------
        input_path : Str, Raw data path.

        """
        # pass input path
        self.load_path = input_path

        # Get general and inner paths
        self.gen_path, innerpath = sep_dir(self.load_path,1)
        
        # load object properties as dict
        obj_props = lab2mat.load(os.path.join(self.gen_path, 'organized.json'))
        
        # get info
        self.ch_struct = obj_props['ch_struct']
        self.win = obj_props['win']
        self.fs = obj_props['fs']
        self.down_factor = obj_props['down_factor'] # downsample factor
        self.animal_ids = obj_props['animal_ids'] # animal IDs in folder
        self.filelist = obj_props['filelist']
        
        # display instance attributes
        pprint(vars(self))
    
  
    # main methods (iterate over files)
    def mainfunc(self):
        """
        main function that iterates over animals, extract comments, and
        saves them in a .csv file for human readable format
        """
        # init progress bar
        total = len(self.filelist)*len(self.animal_ids)       
        
        cntr = 1 # init counter
        # loop through labchart files (multilple animals per file)
        
        # create save list
        self.save_list = []
        for i in range(len(self.filelist)):
            
            # get adi file obj
            f = adi.read_file(os.path.join(self.load_path, self.filelist[i])) 
            
            # get channel list
            ch_idx = np.linspace(1,f.n_channels,f.n_channels,dtype=int)
            ch_list = np.split(ch_idx, round(f.n_channels/len(self.ch_struct)))
            
            # error check for animals vs ch_list for correct allocation of channels
            if len(ch_list) - len(self.animal_ids) != 0:
                print('Animal numbers do not match channel structure')
                
            for ii in range(len(ch_list)): # iterate through animals
         
                # extract and save comments
                self.extact_comments(f,self.animal_ids[ii],ch_list[ii])
                
                print(cntr, 'of',total ,'Experiments Saved')
                cntr += 1 # update counter
                
        # convert to dataframe and save to csv
        df = pd.DataFrame(self.save_list)
        df.to_csv(os.path.join(self.gen_path,'Extracted_seizures.csv'), header = False, index = False)
        

    # save in chunks per animal
    def extact_comments(self, file_obj, filename, ch_list):
        """
        extact_comments(self, file_obj, filename, ch_list)

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
        breakpoint()
        ch_list = ch_list - 1 # convert channel to python format
        all_blocks = len(file_obj.channels[0].n_samples) # get all blocks
        
        for block in range(all_blocks):
            
            # get first channel (applies across animals channels)
            chobj = file_obj.channels[ch_list[0]] # get channel obj
            
            try: # skip corrupted blocks
                chobj.get_data(block+1,start_sample=0,stop_sample=1000)
            except:
                print(block, ' is corrupted')
                continue
            
            # get file name
            file_id  = filename + ascii_lowercase[block] + '.h5' # add extension

            # get comments
            user_coms = file_obj.records[block].comments
            
            # filter coms based on channel id
            coms, com_idx = SzrsFromLab.filter_coms(user_coms, ch_list[0])
                
            # get szr index comments    
            szr_idx = list(np.flatnonzero(np.core.defchararray.find(coms,'ictal')!=-1))
            
            # append to seizure list
            self.save_list.append(szr_idx.insert(0,file_id))        

    @staticmethod
    def filter_coms(comments, channel_id):
        """
        filter_coms(comments, channel_id)

        Parameters
        ----------
        comments : list of comments objects
        channel_id : Int, channel numner

        Returns
        -------
        coms: np array.
        com_time : ndarray, comment times.

        """
        coms = np.array([]) # create comment array
        com_time =  np.array([]) # create comment time array in seconds
        
        for c in comments: # iterate comments
            if c.channel_ == channel_id: # if comment channel matches channel number
                coms = np.append(coms, c.text) # append com
                com_time = np.append(com_time, c.time) # append com time
                
        return np.core.defchararray.lower(coms), com_time
        

# Execute if module runs as main program
if __name__ == '__main__':
    # create instance
    obj = SzrsFromLab(input_path)
   
    # run analysis
    obj.mainfunc()
   
      



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        