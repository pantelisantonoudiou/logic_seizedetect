# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:43:46 2020

@author: Pante
"""
import matplotlib.pyplot as plt
import os,sys
import pandas as pd
import numpy as np
if (os.path.dirname(os.path.abspath(os.getcwd())) in sys.path) == False:
    sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
from array_helper import find_szr_idx, match_szrs, merge_close

paths = [ 
    r'C:\Users\Pante\Desktop\seizure_data_tb\train\model_performance\all_method_metrics_train_1.csv',
    r'C:\Users\Pante\Desktop\seizure_data_tb\test\model_performance\best_method_metrics_1.csv',
    ]

# Get all features 
df = pd.read_csv(paths[0]) # read df
col_idx = df.columns.str.contains('Enabled') # get enabled columns index
features = df.columns[col_idx] # get
features = np.array([x.replace('Enabled_', '') for x in features])

f1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
idx = np.zeros((len(df),len(paths)), dtype=bool)
for i in range(0,len(paths)):
    df = pd.read_csv(paths[i])
    ax1.plot(df['detected_ratio'])
    ax2.plot(df['false_positives'])
    idx[:,i] = np.array((df['detected_ratio'] > 0.99) & (df['false_positives'] < 700))
     

# get common index    
common_idx = np.where(np.logical_and(idx[:,0], idx[:,1]))[0]
df_best = df.loc[common_idx]

for i in range(common_idx.shape[0]): 
    enabled_idx = np.array(df.loc[common_idx[i]][col_idx]).astype(bool)
    print(common_idx[i], features[enabled_idx])

# get false positives    
df_list = []    

for i in range(0,len(paths)):
    df = pd.read_csv(paths[i])
    df_list.append(df['false_positives'][common_idx])
    plt.plot(df_list[i])
    
