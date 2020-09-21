# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:58:54 2020

@author: panton01
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from array_helper import find_szr_idx, merge_close
from multich_dataPrep import lab2mat

# get main path
main_path = r'W:\Maguire Lab\Trina\2020\06- June\5142_5143_5160_5220'


def main_func(main_path):
    
    # dict load
    settings = lab2mat.load(os.path.join(main_path, 'organized.json'))
    
    # get user seizures
    df = pd.read_csv(os.path.join(main_path, 'Extracted_seizures.csv'), header = None)
    
    # get verified predictions file list
    ver_path = os.path.join(main_path, 'verified_predictions_pantelis')
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(ver_path)))
    
    if len(df) != len(filelist): # file check
        print('Warning: length of extracted seizures does not match list of verified predictions!')
    print(str(len(filelist)) + ' files will be analyzed...')
    
    # create dataframe to store metrics
    df_save = pd.DataFrame(np.zeros((len(df)+1,3)), columns = ['total', 'detected', 'false_positives'])
    df_save.insert(0, 'exp_id', filelist + ['Grand_sum'])
    

    for i in tqdm(range(len(filelist))):
        
        # get user scored seizured index (gold standard)
        true_idx = get_szr_index(df, filelist[i].replace('.csv',''))
        
        # load seizure index from user-curated model-predictions 
        rawpred = np.loadtxt(os.path.join(ver_path, filelist[i]), delimiter=',', skiprows=1)
        
        if np.sum(rawpred)>0: # check if any seizures were detected

            # get index bounds of semi-manual detected seizures
            pred_bounds = find_szr_idx(rawpred, np.array([0,1]))
            pred_bounds *= settings['win'] # convert to seconds
            
            # get matching seizures
            df_save['total'].at[i] = true_idx.shape[0] # total
            df_save['detected'].at[i], non_detected_idx = get_match(true_idx, pred_bounds) # detected
            df_save['false_positives'].at[i] = pred_bounds.shape[0] - df_save['detected'][i] # false positives
            
            if df_save['total'][i] != df_save['detected'][i]:
                print('not all seizures were detected')
                print(filelist[i], get_hours(non_detected_idx))
    
    # get grand totals
    df_save['total'].at[i+1] =  df_save['total'].sum()    
    df_save['detected'].at[i+1] =  df_save['detected'].sum()
    df_save['false_positives'].at[i+1] =  df_save['false_positives'].sum() 
    
    # save csv
    df_save.to_csv(os.path.join(main_path, 'detected.csv'),index=False)
    print('Metrics for seizure matching completed')    
        
def get_match(true_idx, pred_bounds):
    """
    get_match(true_idx, pred_bounds)

    Parameters
    ----------
    true_idx : ndarray flat, index of true seizures
    pred_bounds : 2D ndarray, semi-automated seizure bounds (rows = sezuires, cols = [start, stop])

    Returns
    -------
    detected : Int, Number of detected seizures from true idx
    non_detected_idx: ndarray flat, index of non matched seizures

    """

    detected = 0 # set detected to 0
    detected_idx = np.array([],dtype=np.int64) # init empty array
    
    # get all_idx
    all_idx = np.linspace(0,true_idx.shape[0]-1,true_idx.shape[0], dtype=np.int64)
    for i in range(pred_bounds.shape[0]): # iterate through pred seizures
        
        # find if true idx is between predictions low and upper bounds
        idx = (pred_bounds[i,0] < true_idx) & (pred_bounds[i,1] > true_idx)
        
        if any(idx): # if detected add 1
            detected += 1 
            detected_idx = np.append(detected_idx, np.where(idx))  # append to detected idx
    
    # get only non detected index
    non_detected_idx = np.delete(all_idx, detected_idx)
        
    return detected, true_idx[non_detected_idx] 

    
def get_szr_index(df, exp_id):
    """
    idx = get_szr_index(df, exp_id)
    
    Returns seizure index from dataframe when given experiment ID

    Parameters
    ----------
    df : dataframe containing seizure index with exp id
    exp_id : string, experiment identifier

    Returns
    -------
    idx : flattened ndarray, seizure index in seconds

    """
    
    # get row that matches experiment ID
    idx = np.array(df[df.iloc[:,0].str.match(exp_id)]).flatten()
    
    # get only numbers as float
    idx = idx[1:].astype(np.float64)
    
    # drop nans
    idx = idx[np.logical_not(np.isnan(idx))]
    
    return idx
   
   
def get_hours(seconds_list):
    """
    get_hours(seconds_list)

    Parameters
    ----------
    seconds_list : seconds array

    Returns
    -------
    str1 : list with times in hour format

    """
    str1 = []
    for seconds in list(seconds_list):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        str1.append('{}:{}:{}'.format(int(hours), int(minutes), int(seconds)))
    return (str1) 
    
# Execute if module runs as main program
if __name__ == '__main__':
    main_func(main_path)






















