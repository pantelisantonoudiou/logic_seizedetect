# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:23:46 2020

@author: Pante
"""


import os, features, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set channel names
ch_dict = {'0': 'vhpc','1':'fc'}

# set main path
main_path = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\szr_catalogue_mean'

# get file list of seizure catalogues
filelist = list(filter(lambda k: '.csv' in k, os.listdir(main_path)))


# cols = ['-120_-90', '-90_-60', '-60_-30', '-30_0', 'during_szr','0_30',
#         '30_60', '60_90', '90_120']


# cols = ['-120_-90', '-90_-60', '-60_-30', '-30_0', '0_30',
#         '30_60', '60_90', '90_120']

# for i in range(len(filelist)): # len(filelist)
    
#     # read dataframe
#     df = pd.read_csv(os.path.join(main_path, filelist[i])) 
    
#     # select data
#     data = df[cols]
    
#     #plot
#     plt.figure()
#     sns.boxplot(data= data)
#     plt.xlabel('Time (seconds)')
#     plt.title(filelist[i])
    
 
cols = ['x_sdevs'] # ['szr_percentile'] ['x_sdevs']
df = pd.read_csv(os.path.join(main_path, filelist[0])) 
box_data = pd.DataFrame(data = np.zeros((len(df),0)))
for i in range(len(filelist)): # len(filelist)
    
    # read dataframe
    df = pd.read_csv(os.path.join(main_path, filelist[i])) 
    
    # get name
    col_name = filelist[i][:-4]
    if any(char.isdigit() for char in col_name) is True:
    #     col_name = col_name.split('_')[-1]
    #     col_name = col_name[0] + '_' + ch_dict[col_name[1]]
    
        # insert columns
        box_data.insert(loc=0, column = col_name, value = df[cols])
    
# plot box plot
plt.figure()
ax = sns.boxplot(data = box_data)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.xlabel('Time (seconds)')
plt.ylabel(cols)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    