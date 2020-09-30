# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:00:40 2020

@author: Pante
"""


import os, sys
import pandas as pd
import numpy as np

# path to comapre files
paths = [ 
    'train/model_performance/all_method_metrics_train_1.csv',
    'test/model_performance/best_method_metrics_1.csv',
         ]

def get_common(main_path, detected_threshold = 0.99, fp_threshold = 700):
    """
    get_common(main_path, detected_threshold = 0.99, fp_threshold = 700)
    
    Save common metrics according to thresholds

    Parameters
    ----------
    main_path : Str
    detected_threshold : Float, optional, minimum detected ratio threhsold
    fp_threshold : Float, optional, max false positive threhsold

    Returns
    -------
    None.
    """
    
    # Get all features 
    df = pd.read_csv(os.path.join(main_path, paths[0])) # read df
    idx = np.zeros((len(df),len(paths)), dtype=bool) # create index
    
    for i in range(len(paths)): # iterate through paths
        
        # load dataframe
        df = pd.read_csv(os.path.join(main_path, paths[i]))
        
        # get index
        idx[:,i] = np.array((df['detected_ratio'] > detected_threshold) & (df['false_positives'] < fp_threshold))

    # Save df with filtered common index to csv   
    common_idx = np.where(np.logical_and(idx[:,0], idx[:,1]))[0] # get common idx
    df_best = df.loc[common_idx] # get df with common index
    df_best.to_csv(os.path.join(main_path,'methods_table.csv'), index = False) # save csv

if __name__ == '__main__':
    if len(sys.argv) == 2:
        get_common(sys.argv[1]) # instantiate and pass main path
    else:
        print('Please provide parent directory')

# Original Plots
#
#
# 
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# import numpy as np
#
# # Get all features 
# df = pd.read_csv(paths[0]) # read df
# col_idx = df.columns.str.contains('Enabled') # get enabled columns index
# features = df.columns[col_idx] # get
# features = np.array([x.replace('Enabled_', '') for x in features])

# f1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# idx = np.zeros((len(df),len(paths)), dtype=bool)
# for i in range(0,len(paths)):
#     df = pd.read_csv(paths[i])
#     ax1.plot(df['detected_ratio'])
#     ax2.plot(df['false_positives'])
#     idx[:,i] = np.array((df['detected_ratio'] > 0.99) & (df['false_positives'] < 700))
     

# # get common index    
# common_idx = np.where(np.logical_and(idx[:,0], idx[:,1]))[0]
# for i in range(common_idx.shape[0]): 
#     enabled_idx = np.array(df.loc[common_idx[i]][col_idx]).astype(bool)
#     print(common_idx[i], cols[enabled_idx])

# # get false positives    
# df_list = []    

# for i in range(0,len(paths)):
#     df = pd.read_csv(paths[i])
#     df_list.append(df['false_positives'][common_idx])
#     plt.plot(df_list[i])
    
# df_best2 = df.loc[common_idx]