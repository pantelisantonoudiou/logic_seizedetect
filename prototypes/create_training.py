# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:55:09 2020

@author: Pante
"""

import os, tables, time, pdb
import numpy as np
from tqdm import tqdm
from numba import jit
from data_augment import sliding_window_multi
from array_helper import find_szr_idx
# import matplotlib.pyplot as plt

@jit(nopython=True)  
def generate_list(n,idx_excl,n_rand):
    """
    random_array = generate_list(n,idx_excl,n_rand)
    Generates a list

    Parameters
    ----------
    n : INT, total number of samples
    idx_excl : 2D array, start,stop index for each seizure
    n_rand : INT, number of random samples

    Returns
    -------
    1D array index for selection of non-seizure events

    """
    
    # get all samples in an array
    full_array = np.linspace(0,n,n-1); full_array = full_array.astype(np.int64)
    
    # create empty array to add seizure index
    exc_array = np.empty((0),dtype = np.int64);
    
    for i in range(idx_excl.shape[0]):
        start = idx_excl[i,0]; stop = idx_excl[i,1] # get stop and start boundaries
        temp_array = np.linspace(start, stop, (stop-start)+1); temp_array = temp_array.astype(np.int64)
        exc_array = np.concatenate((exc_array,temp_array), axis = 0)
    
    # remove seizure and surounding elements
    full_array = np.delete(full_array, exc_array)
    
    # return random choice
    return np.sort(np.random.choice(full_array,n_rand))


class CreateTrain():
    """
    Create training data for machine learning
    """
    
    
    def __init__(self, main_path, win, fs, chnls):
        """
        Class constructor (data retrieval)

        Parameters
        ----------
        main_path : str
        win : int, window of seizure traces
        fs : int, sampling rate
        chnls : INT, channel number

        """
        
        self.win = win # window size
        self.fs = fs # sampling rate
        self.chnls = chnls # channels
        
        self.samples_added = 0 # print number of samples added
        
        self.verpred_path = os.path.join(main_path, 'verified_predictions_pantelis')
        self.data_path = os.path.join(main_path, 'reorganized_data')
        
        # get filelist of verified predictions
        filelist_ver = list(filter(lambda k: '.csv' in k, os.listdir(self.verpred_path)))
        filelist_ver = [os.path.splitext(x)[0] for x in filelist_ver] # remove ending
        
        # get filelst of organized data
        filelist_data = list(filter(lambda k: '.h5' in k, os.listdir( self.data_path)))
        filelist_data = [os.path.splitext(x)[0] for x in filelist_data] # remove ending
        
        # get list of common elements between verified predictions and data files
        self.filelist = [x for x in filelist_data if x in filelist_ver]
        
    
    
    def get_data(self, data, idx_bounds):
        """
        # get x and y data for one experiment

        Parameters
        ----------
        data : 2D array (voltage), 1D = time, 2d = channels
        idx_bounds : TYPE

        Returns
        -------
        x_data : 3D array (voltage), 1d = segments, 2d = time, 3d = channels
        y_data : 1D array, binary labels 1 =s eizure, 0 = non seizure

        """
        trim_bounds = 2 # segment to reduce user defined seizure bounds
        szr_win_slide_ratio = 0.1 # ratio for sliding window for seizures
        nonszr_to_szr_ratio = 3 # non seizure to seizure ratio
        
        # get new index with +/- 5 mins around seizures
        idx_bounds_expanded = np.copy(idx_bounds)
        idx_bounds_expanded[:,0] -= int(5*60/self.win)
        idx_bounds_expanded[:,1] += int(5*60/self.win)
        
        # init train data per experiment
        x_data = np.empty((0, int(self.win*self.fs), self.chnls))
        x_data = x_data.astype(np.float64)
        
        # print('Extracting training segments from:', self.filelist[i],'...')           
        for ii in range(idx_bounds.shape[0]): # loop through seizures
            
            # get continous seizure segments across channels
            szr_segment = data[idx_bounds[ii,0]+trim_bounds:idx_bounds[ii,1]-trim_bounds,:,:].reshape((-1, data.shape[2]))
            
            # augument by window sliding
            szr_mat = sliding_window_multi(szr_segment, int(self.win*self.fs), szr_win_slide_ratio)
            
            # concatenate train data
            x_data = np.concatenate((x_data, szr_mat), axis = 0)
        
        # get non seizure index and segments
        non_szr_idx = generate_list(data.shape[0]-1, idx_bounds_expanded, x_data.shape[0]*nonszr_to_szr_ratio) # index
        non_szr_mat = data[non_szr_idx,:,:] # get segments from data
        
        # create ground truth bool for szr and non-szr segments
        y_data = np.ones(x_data.shape[0], dtype = np.int64)
        y_data = np.concatenate((y_data, np.zeros(non_szr_mat.shape[0], dtype = np.int64)))
        
        # add non seizure data
        x_data = np.concatenate((x_data, non_szr_mat), axis = 0)
        
        return x_data, y_data
      
    
    def save_func(self):
        '''
        Save training dataset to pytables datastore for one folder

        '''
        
        # Saving Parameters
        atom = tables.Float64Atom() # declare data type 
        fsave_xdata = tables.open_file(os.path.join(main_path, 'x_data.h5') , mode = 'w') # open tables object
        fsave_ytrue = tables.open_file(os.path.join(main_path, 'y_data.h5') , mode = 'w') # open tables object
        ds_x = fsave_xdata.create_earray(fsave_xdata.root, 'data',atom, # create data store 
                                    [0, int(self.win*self.fs), self.chnls])
        ds_y = fsave_ytrue.create_earray(fsave_ytrue.root, 'data',atom, # create data store 
                                    [0])
        
        for i in tqdm(range(len(self.filelist))): # loop through files
              
            # load data
            f = tables.open_file(os.path.join(self.data_path, self.filelist[i]+'.h5'), mode='r')
            data = f.root.data[:]; f.close()
            
            # Get ground truth data
            y = np.loadtxt(os.path.join(self.verpred_path, self.filelist[i]+'.csv'), delimiter=',', skiprows = 0)

            # find seizure segments    
            idx_bounds = find_szr_idx(y, np.array([0,1], dtype=int))    
            
            if idx_bounds.shape[0] > 0: # if seizures were detected     
                
                # get x and y data
                x_data, y_data = self.get_data(data, idx_bounds) #  
                # x_data, y_data = get_data_static(self.win,self.fs,self.chnls, data, idx_bounds) #  
      
                # append x and y data to datastore
                ds_x.append(x_data)
                ds_y.append(y_data)
                
        # close save objects    
        fsave_xdata.close()
        fsave_ytrue.close() 
        print('Training dataset created.')
        
        
   # create training data and append to datastore
    def append_func(self, ds_x, ds_y):
        '''
        Append training dataset to pytables datastore for one folder

        '''
        
        for i in range(len(self.filelist)): # loop through files
            print('extracting data from', self.filelist[i])
        
            # load data
            f = tables.open_file(os.path.join(self.data_path, self.filelist[i]+'.h5'), mode='r')
            data = f.root.data[:]; f.close()
            
            # Get ground truth data
            y = np.loadtxt(os.path.join(self.verpred_path, self.filelist[i]+'.csv'), delimiter=',', skiprows = 0)

            # find seizure segments    
            idx_bounds = find_szr_idx(y, np.array([0,1], dtype=int))    
            
            if idx_bounds.shape[0] > 0: # if seizures were detected     
                
                # get x and y data
                x_data, y_data = self.get_data(data, idx_bounds) #  
                # x_data, y_data = get_data_static(self.win,self.fs,self.chnls, data, idx_bounds) #
                self.samples_added+= y_data.shape[0]
                print(self.samples_added,'samples added')
                # append x and y data to datastore
                ds_x.append(x_data)
                ds_y.append(y_data)
                

# main function for training data generation
def multi_folder(main_path, win, fs, chnls):
    """
    multi_folder(main_path, win, fs, chnls)
    Save training dataset for multiple folders

    Parameters
    ----------
    main_path : str
    win : int, window of seizure traces
    fs : int, sampling rate
    chnls : INT, channel number

    """
    
    # initialize saving Parameters
    atom = tables.Float64Atom() # declare data type 
    fsave_xdata = tables.open_file(os.path.join(main_path, 'x_data.h5') , mode = 'w') # open tables object
    fsave_ytrue = tables.open_file(os.path.join(main_path, 'y_data.h5') , mode = 'w') # open tables object
    ds_x = fsave_xdata.create_earray(fsave_xdata.root, 'data',atom, # create data store 
                                [0, int(win*fs), chnls])
    ds_y = fsave_ytrue.create_earray(fsave_ytrue.root, 'data',atom, # create data store 
                                [0])
    
    # get all child directory paths
    paths = [f.path for f in os.scandir(main_path) if f.is_dir()]
    
    for path in paths: # loop though paths
        
        # init instance
        obj = CreateTrain(path, win, fs, chnls)
        
        # append to datastore
        obj.append_func(ds_x,ds_y)
        
        print ('------------------')
        print(path,'completed.')
         
    # close save objects    
    fsave_xdata.close()
    fsave_ytrue.close() 
    print('Training dataset created.')

# Execute if module runs as main program
if __name__ == '__main__':
    
    # set input path
    main_path = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'
    
    #### For multiple folders folder #####
    tic = time.time()
    
    # init object
    obj = multi_folder(main_path, win = 5, fs = 100, chnls = 3)
    
    print('time elapsed:',time.time()-tic ,'seconds')
    #### ----------------- #####
    
    
     # #### For one folder #####
    
    # tic = time.time()
    
    # # init object
    # obj = CreateTrain(main_path,5,100,3)
    
    # # get predictions in binary format and store in csv
    # obj.save_func()

    # print('time elapsed:',time.time()-tic ,'seconds')
    
    
    #   #### ----------------- #####
    
    
    
    
    
    
    
    
    
    