# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:58:54 2020

@author: panton01
"""

import os
import numpy as np
import pandas as pd

# get main path
main_path = r'W:\Maguire Lab\Trina\2020\06- June\5142_5143_5160_5220'


def main_func(main_path):
    
    # get user seizures
    df = pd.read_csv(os.path.join(main_path, 'Extracted_seizures.csv'), header = None)
    
    # get verified predictions file list
    filelist = list(filter(lambda k: '.csv' in k, os.listdir(os.path.join(main_path, 'verified_predictions_pantelis'))))
    
    if len(df) != len(filelist): # file check
        print('Warning: length of extracted seizures does not match list of verified predictions!')
    
    for i in range(len(filelist)):
        
        # get user scored seizured index
        idx = get_szr_index(df, filelist[i].replace('.csv',''))
        
        # load predictions
        
        
        

    
    
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
    
    
# Execute if module runs as main program
if __name__ == '__main__':
    main_func(main_path)






















