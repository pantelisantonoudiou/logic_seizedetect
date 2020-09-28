# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:02:29 2020

@author: Pante
"""

### ------------------- Imports ------------------- ###
import os, tables
import numpy as np
### ----------------------------------------------- ###

def get_data(main_path, exp_path, ch_num = [0,1] , inner_path=[], load_y = True):
    """
    get_data(main_path, exp_path, ch_num = [0,1] , inner_path=[], load_y = True)
    get_data(main_path, exp_path, ch_num = [0,1] , inner_path={'data_path':'filt_data'}, load_y = False)

    Parameters
    ----------
    main_path : Str
    exp_path : Str
    ch_num : List, channels to be loaded, optional, The default is [0,1].
    inner_path : Define inner folder for x or/and y data loading, optional The default is empty list.
    load_y : Bool, If False only data are returned, optional, The default is True.

    Returns
    -------
    data : ndarray (1d = segments, 2d = time, 3d = channels)
    y_data : 1D Int ndarray

    """
    
    if len(inner_path) == 0:
        inner_path = {'data_path':'reorganized_data', 'pred_path':'verified_predictions_pantelis'}
    
    # load lfp/eeg data
    f = tables.open_file(os.path.join(main_path, inner_path['data_path'], exp_path+'.h5'), mode = 'r') # open tables object
    data = f.root.data[:]; f.close() # load data
    data = data[:,:,ch_num] # get only desirer channels
    
    y_data = []
    if load_y == True: # get ground truth labels
        y_data = np.loadtxt(os.path.join(main_path, inner_path['pred_path'], exp_path+'.csv'), delimiter=',', skiprows=0)
        y_data = y_data.astype(np.bool) # conver to bool
        return data, y_data
    
    return data


def save_data(main_path, exp_path, data):
    """
    save_data(main_path, exp_path, data)

    Parameters
    ----------
    main_path : Str
    exp_path : Str
    data : ndarray

    Returns
    -------

    """
    
    try:
        # Saving Parameters
        atom = tables.Float64Atom() # declare data type 
        fsave = tables.open_file(os.path.join(main_path, exp_path+'.h5') , mode = 'w') # open tables object
        ds = fsave.create_earray(fsave.root, 'data', atom, # create data store 
                                    [0, data.shape[1], data.shape[2]])
        ds.append(data) # append data
        fsave.close() # close tables object
        return 1
    
    except:
        print('File could not be saved')
        return 0
    
# get one channel features metrics 
def get_features_single_ch(data,param_list):
    """
    get_features_single_ch(data,param_list)

    Parameters
    ----------
    data : 2D ndarray (1d = segments, 2d = time)
    param_list : ndarray with functions

    Returns
    -------
    x_data : 2D ndarray (rows = segments, columns = features)
    feature_labels : List
    """
    
    # Init data matrix
    x_data = np.zeros((data.shape[0], param_list.shape[0]))
    x_data = x_data.astype(np.double)
    
    feature_labels = [] # init labels list.
    for ii in range(param_list.shape[0]): # iterate over parameter list
    
        # append function name (feature) to list   
        feature_labels.append(param_list[ii].__name__)
        
        for i in range(data.shape[0]):    # iterate over segments
            x_data[i,ii] = param_list[ii](data[i,:]) # extract feature
            
    return x_data, feature_labels

# get cross-channel feature metrics
def get_features_crossch(data, param_list):
    """
    get_features_crossch(data, param_list)
    
    Parameters
    ----------
    data : 3D ndarray (1d = segments, 2d = time, 3d = channels)
    cross_ch_param_list : ndarray with functions

    Returns
    -------
    x_data : 2D ndarray (rows = segments, columns = features)
    feature_labels : List
    """
    
    # Init data matrix
    x_data = np.zeros((data.shape[0], param_list.shape[0]))
    x_data = x_data.astype(np.double)
    
    feature_labels = [] # init labels list
    for ii in range(param_list.shape[0]): # iterate over parameter list
    
        # append function name (feature) to list      
        feature_labels.append(param_list[ii].__name__)
        
        for i in range(data.shape[0]):    # iterate over segments
            x_data[i,ii] = param_list[ii](data[i,:,0], data[i,:,1]) # extract feature
        
    return x_data, feature_labels
    

# get features for all channels
def get_features_allch(data,param_list,cross_ch_param_list):
    """
    get_features_allch(data,param_list,cross_ch_param_list)

    Parameters
    ----------
    data : 3D ndarray (1d = segments, 2d = time, 3d = channels)
    param_list : ndarray with functions

    Returns
    -------
    x_data :  2D ndarray (rows = segments, columns = features)
    labels : np.array, feature names
    """
    labels = [] # make list to store labels
    x_data = np.zeros((data.shape[0],0),dtype=np.float) # make array to store all features
    
    # calculate single channel measures
    for ii in range(data.shape[2]):
        # get data and features labels per channel
        temp_data, feature_labels = get_features_single_ch(data[:,:,ii], np.array(param_list))
        x_data = np.concatenate((x_data, temp_data), axis=1) # append data
        
        str2append = '_' + str(ii) # get channel string
        labels += [s + str2append for s in feature_labels] # add to feature labels
    
    # calculcate cross-channel measures
    if len(cross_ch_param_list)>0:
        temp_data, feature_labels = get_features_crossch(data, np.array(cross_ch_param_list))
        x_data = np.concatenate((x_data, temp_data), axis=1) # append data
        labels += [s for s in feature_labels] # add to feature labels   
     
    return x_data, np.array(labels)


