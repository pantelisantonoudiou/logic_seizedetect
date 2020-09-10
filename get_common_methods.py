# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:00:40 2020

@author: Pante
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

paths = [r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\model_performance\all_method_metrics_train_2.csv',
         r'C:\Users\Pante\Desktop\seizure_data_tb\Test_data\model_performance\best_method_metrics_2.csv'
         ]

def get_best(paths):
         
df = pd.read_csv(paths[0])
col_idx = df.columns.str.contains('Enabled')
cols = df.columns[col_idx]
cols = np.array([x.replace('Enabled_', '') for x in cols])

f1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
idx = np.zeros((len(df),len(paths)), dtype=bool)
for i in range(0,len(paths)):
    df = pd.read_csv(paths[i])
    ax1.plot(df['detected_ratio'])
    ax2.plot(df['false_positives'])
    idx[:,i] = np.array((df['detected_ratio'] > 0.99) & (df['false_positives'] < 700))
     

# get common index    
common_idx = np.where(np.logical_and(idx[:,0], idx[:,1]))[0]
for i in range(common_idx.shape[0]): 
    enabled_idx = np.array(df.loc[common_idx[i]][col_idx]).astype(bool)
    print(common_idx[i], cols[enabled_idx])

# get false positives    
df_list = []    

for i in range(0,len(paths)):
    df = pd.read_csv(paths[i])
    df_list.append(df['false_positives'][common_idx])
    plt.plot(df_list[i])
    
df_best2 = df.loc[common_idx]
