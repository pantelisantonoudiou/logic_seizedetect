# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:23:46 2020

@author: Pante
"""


import os, features, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # set channel names
# ch_dict = {'0': 'vhpc','1':'fc'}

# # set main path
# main_path = r'C:\Users\Pante\Desktop\seizure_data_tb\szr_catalogue_mean_train'

# # get file list of seizure catalogues
# filelist = list(filter(lambda k: '.csv' in k, os.listdir(main_path)))


# # define time bins
# cols = ['-120_-90', '-90_-60', '-60_-30', '-30_0', 'during_szr','0_30',
#         '30_60', '60_90', '90_120']
## cols = ['-120_-90', '-90_-60', '-60_-30', '-30_0', '0_30',
##         '30_60', '60_90', '90_120']

# ## #1 How values vary with time
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
 

# ### #2 probability of a seizure being smaller than surrounding regions
# anot_heat = pd.DataFrame(data = np.zeros( (len(filelist),len(cols)) ), columns = cols)
# for i in range(len(filelist)):
    
#     # read dataframe
#     df = pd.read_csv(os.path.join(main_path, filelist[i])) 
    
#     # select data
#     col_name = filelist[i][:-4]
#     if col_name[-1].isdigit() is True:
#         for ii in range(len(cols)):
#             anot_heat.iloc[i, ii] = sum(df['during_szr'] < df[cols[ii]])  

# anot_heat/=len(filelist) # convert to probability
# anot_heat = anot_heat.round(2)
# anot_heat.insert(loc = 0, column = 'feature', value = filelist)


# ### #3 how much bigger is the seizure
# for i in range(len(filelist)): #len(filelist)
    
#     # read dataframe
#     df = pd.read_csv(os.path.join(main_path, filelist[i])) 
    
#     data = np.array(df[cols])
#     data = np.divide(data.T, np.array(df['during_szr']))
    
#     # plot
#     plt.figure()
#     sns.boxplot(data= pd.DataFrame(data = data.T, columns = cols))
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Relative differnce')
#     plt.title(filelist[i])
    

# ####  #4
# cols = ['x_sdevs'] # ['szr_percentile'] ['x_sdevs']
# df = pd.read_csv(os.path.join(main_path, 'line_length_0.csv')) 
# idx = df[cols] < 20
# box_data = pd.DataFrame(data = np.zeros((len(df),0)))
# for i in range(len(filelist)): 
    
#     # read dataframe
#     df = pd.read_csv(os.path.join(main_path, filelist[i])) 
    
#     # get name
#     col_name = filelist[i][:-4]
#     # if col_name[-1].isdigit() is True:
#         # if int(col_name[-1]) == 0:
#         #     col_name = col_name[:-1] + ch_dict[col_name[-1]] # remap name
    
#     # insert columns
#     box_data.insert(loc=0, column = col_name, value = df[cols][idx]) # [idx]
    
# # plot box plot
# plt.figure()
# ax = sns.boxplot(data = box_data)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
# plt.xlabel('Time (seconds)')
# plt.ylabel(cols)
# plt.title(ch_dict)
     
    
## # plot correlation    
## sns.heatmap(box_data.corr(), 
##         xticklabels=box_data.columns,
##         yticklabels=box_data.columns)    
  

###### ---------------------- SELECT DATA BASED ON COST AND FEATURE METRICS  ----------------------------- ######  
# feature_list =[] # init feature list

##### ------------------  GET ALSO OPTIMUM THRESHOLD ----------------------------- #####   
      
# get subdirectories
main_path = r'C:\Users\Pante\Desktop\seizure_data_tb\optimum_threshold'
filelist = list(filter(lambda k: '.csv' in k, os.listdir(main_path))) # get only files with predictions

# get prototype dataframe
df = pd.read_csv(os.path.join(main_path,filelist[0]))

# get thresholds
thresh_array = np.zeros(len(filelist))
for i in range(len(filelist)):
    
    # isolate threshold level digits
    threshold = filelist[i].split('_')[1][:-4].split('-')
    thresh_array[i] = int(threshold[0]) + int(threshold[1])/10 # reconstruct reholds

# init arrays
max_detected = np.zeros(len(df))
min_falsepos = np.zeros(len(df))
optimum_threshold = np.zeros(len(df))
selected_cost = np.zeros(len(df))

# create dataframe to store ranks
df_rank = pd.DataFrame(data = np.zeros((4, len(df))), columns = df['features'])
df_logic = pd.DataFrame(data = np.zeros((4, len(df))), columns = df['features'])
for ii in range(len(df)): 
    
    # init empty arrays
    detected_ratio = np.zeros(len(filelist))
    false_positives = np.zeros(len(filelist))
    
    for i in range(len(filelist)):
        
        # load dataframe
        df = pd.read_csv(os.path.join(main_path,filelist[i]))
        
        # get detected ratio and false positives for specific feature
        detected_ratio[i] = df['detected_ratio'][df['features'] == df['features'][ii]]
        false_positives[i] = df['false_positives'][df['features'] == df['features'][ii]]
    
    detected_ratio *= 100 # convert to percentage
    
    #### ------ 1 get max value ------------------- ####
    # max_val = np.max(detected_ratio)
    # max_idx = np.where(detected_ratio == max_val)[0]
    
    # alpha = 0.98
    # if max_val > alpha:
    #     max_idx = np.where(detected_ratio>alpha)[0]
        
    ### ------ 2 or get minimum cost --------------- ####
    cost = np.log(false_positives) - detected_ratio
    max_idx = [np.argmin(cost)]
    
    # get minimum false positives from max detected
    idx = np.argmin(false_positives[max_idx])
    idx = max_idx[idx] # remap to original index
    
    max_detected[ii] = detected_ratio[idx]
    min_falsepos[ii] = false_positives[idx]
    optimum_threshold[ii] = thresh_array[idx]
    selected_cost[ii] = cost[idx]
    
# select data
# feature_list.extend(list(df['features'][selected_cost < np.median(selected_cost)]))
df_rank.iloc[0] = 1/selected_cost
df_logic.iloc[0] = selected_cost < np.median(selected_cost)

fig, (ax1,ax2,ax3) = plt.subplots(3, 1, sharex=True)   
ax1.plot(df['features'], max_detected ,'-o')
ax1.set_ylabel('% seizures detected')
ax1.set_xticklabels(df['features'], rotation= 90)

ax2.plot(df['features'], min_falsepos ,'-o', color='orange')
ax2.set_ylabel('False positives')
ax2.set_xticklabels(df['features'], rotation= 90)

ax3.plot(df['features'], optimum_threshold ,'-o', color = 'green')
ax3.set_ylabel('Threshold (x SD)')
ax3.set_xticklabels(df['features'], rotation= 90)
    
        
### ------------------------------------------------------------------- ###



### ------------------- Get feature metrics ---------------------------- ###
# set main path
main_path = r'C:\Users\Pante\Desktop\seizure_data_tb\feature_selection'

# get file list of seizure catalogues
filelist = list(filter(lambda k: '.csv' in k, os.listdir(main_path)))

for i in range(len(filelist)):
    
    # read dataframe
    df = pd.read_csv(os.path.join(main_path, filelist[i])) 
    
    # select data
    data = df.drop(['exp_id'],axis=1)
    
    df_rank.iloc[i+1] = np.abs(data.median())
    df_logic.iloc[i+1] = data.median()>np.mean(data.median())
    
    #plot
    plt.figure()
    ax = sns.boxplot(data= data)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.xlabel('Time (seconds)')
    plt.title(filelist[i])    
    
### ------------------------------------------------------------------- ###
   
### PLOT SORTED RANKS ###
# df_rank = df_rank.rank(axis=1)
# cols = np.array(df.columns[1:])
# idx = np.argsort(df_rank.mean())
# Plot sorted ranks
# f2 = plt.figure()  
# ax = f2.add_subplot(111) 
# xticklabels = list(cols[idx])
# ranks = np.array(df_rank.mean())
# plt.plot(xticklabels, ranks[idx])
# ax.set_xticklabels(xticklabels,rotation=90) 
    
# create df to save essential info
df_save = pd.DataFrame(data = np.zeros((2,len(data.columns))), columns = data.columns )
df_save.iloc[0] =  df_rank.rank(axis=1).mean() # ranks
df_save.iloc[1] = optimum_threshold # optimum_threshold
df_save.insert(loc = 0, column ='metrics', value = ['ranks', 'optimum_threshold'])
path = Path(main_path)
df_save.to_csv(os.path.join(path.parent, 'feature_properties.csv'), index = False)
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    