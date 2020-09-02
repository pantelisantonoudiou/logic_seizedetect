# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:20:26 2020

@author: Pante
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

paths = [r'C:\Users\Pante\Desktop\seizure_data_tb\Test_data\model_performance\all_method_metrics_cluster.csv',
         r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\model_performance\all_method_metrics_cluster.csv'
         ]
         

for i in range(0,len(paths)):
    df = pd.read_csv(paths[i])
    idx = np.where(df,(df['detected_ratio'] == 1.0) & (df['false_positives'] <150))
    
    cols = set(df.columns)
    cols = np.array([x.replace('Enabled_', '') for x in cols])
    for ii in range(idx.shape[0]):
        col_idx = df.columns.str.contains('Enabled')
        enabled_idx = df.loc[idx[ii]]
        print(cols[col_idx])


# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(df['detected_ratio'])
# ax2.plot(df['false_positives'])



df_best = df[idx]