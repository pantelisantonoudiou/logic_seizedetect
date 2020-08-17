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

file_path = r'C:\Users\Pante\Desktop\seizure_data_tb\line_length_0.csv'

df = pd.read_csv(file_path)
# df.boxplot(column = ['during_szr','-120_-90', '-90_-60', '-60_-30', '-30_0', '0_30',
#        '30_60', '60_90', '90_120'])

cols = ['-120_-90', '-90_-60', '-60_-30', '-30_0', 'during_szr','0_30',
       '30_60', '60_90', '90_120']
data = df[cols]
 
sns.boxplot(data= data) 