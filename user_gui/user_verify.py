# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:10:48 2020

@author: panton01
"""

## ------>>>>> USER INPUT <<<<<< --------------
input_path = r'C:\Users\panton01\Desktop\08-August\5394_5388_5390_5391'
file_id = '082820_5390a.csv' # 5221 5222 5223 5162
enable = 1 # set to 1 to select from files that have not been analyzed
execute = 1 # 1 to run gui, 0 for verification
# list(filter(lambda k: '.h5' in k, os.listdir(obj.rawdata_path)))
## -------<<<<<<<<<<

### -------- IMPORTS ---------- ###
import os, sys, json, tables
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector, TextBox
# User Defined
parent_path = os.path.dirname(os.path.abspath(os.getcwd()))
if ( os.path.join(parent_path,'helper') in sys.path) == False:
    sys.path.extend([parent_path, os.path.join(parent_path,'helper'),
                     os.path.join(parent_path,'data_preparation')])
from multich_data_prep import Lab2Mat
from array_helper import find_szr_idx
### ------------------------------------------ ####

       
class UserVerify:
    """
    Class for User verification of detected seizures
    
    """
    
    # class constructor (data retrieval)
    def __init__(self, input_path):
        """
        lab2mat(main_path)

        Parameters
        ----------
        input_path : Str, Path to raw data.

        """
        # pass general path (parent)
        self.gen_path = input_path
        
        # load object properties as dict
        jsonpath = os.path.join(self.gen_path, 'organized.json') # name of dictionary where propeties are stored
        obj_props = Lab2Mat.load(jsonpath)
        
        # get data path
        self.data_path = os.path.join(self.gen_path, obj_props['org_rawpath'], file_id.replace('.csv','.h5') )
        
        # get raw prediction path
        self.pred_path = os.path.join(self.gen_path, obj_props['rawpred_path'], file_id) # rawpred_path verpred_path
        
        # create user verified path
        verpred_path = 'verified_predictions'
        obj_props.update({'verpred_path' : verpred_path})
        self.verpred_path = os.path.join(self.gen_path, verpred_path)
        
        # write attributes to json file using a dict
        open(jsonpath, 'w').write(json.dumps(obj_props))
        
        # make path if it doesn't exist
        if os.path.exists( self.verpred_path) is False:
            os.mkdir( self.verpred_path)

        # get sampling rate
        self.fs = round(obj_props['fs'] / obj_props['down_factor'])
        
        # get win in seconds
        self.win = obj_props['win'] 

    def main_func(self, filename):
        """
        main_func(self, filename)

        Parameters
        ----------
        filename : String

        Returns
        -------
        data : 3d Numpy Array (1D = segments, 2D = time, 3D = channel)
        idx_bounds : 2D Numpy Array (rows = seizures, cols = start and end points of detected seizures)

        """
        
        print('File being analyzed: ', file_id)

        # get data and predictions
        bin_pred = np.loadtxt(self.pred_path, delimiter=',', skiprows=0)
        idx_bounds = find_szr_idx(bin_pred, np.array([0,1]))
        
        ## ADD refine seizures?
        # bounds_pred = self.refine_based_on_surround(x_data[:,idx], bounds_pred)   
           
        # load raw data for visualization
        f = tables.open_file(self.data_path, mode='r')
        data = f.root.data[:]
        f.close()
        
        # check whether to continue
        print('>>>>',idx_bounds.shape[0] ,'seizures detected')
        
        return data, idx_bounds
    
    def refine_based_on_surround(self, feature, idx):
        """
        refine_based_on_surround(self,data,idx)

        Parameters
        ----------
        feature : 1D Numpy Array, feature over time bins
        idx : 2 column Numpy Array (rows = seizure segments)
            First column = seizure start segment.
            Second column = seizure end segment.

        Returns
        -------
        idx : Same as input index
            Only seizure segments that obey threshold condition are kept.

        """

        # Set Parameters
        logic_idx = np.zeros(idx.shape[0], dtype=int) # pre-allocate logic vector

        for i in range(idx.shape[0]): # iterate through seizure segments
            
            # get seizure and surrounding segments
            bef = feature[idx[i,0] - round(90/self.win) : idx[i,0] - round(30/self.win) + 1] # segment before seizure
            szr = feature[idx[i,0] : idx[i,1]+1] # seizure segment 

            # check if power difference meets threshold
            cond = ( np.abs(np.median(szr)) / np.abs(np.median(bef)) )*100
            if cond > 130:
                logic_idx[i] = 1
        
        # get index that passed threshold
        idx = idx[logic_idx == 1,:]
        
        return idx
        
               
    def save_emptyidx(self,data_len):
         """
         Save user predictions to csv file as binary
        
         Returns
         -------
         None.
        
         """
         # pre allocate file with zeros
         ver_pred = np.zeros(data_len)
         
         # save file
         np.savetxt(os.path.join(self.verpred_path,file_id), ver_pred, delimiter=',',fmt='%i')
         print('Verified predictions for ', file_id, ' were saved')   
            
        
# Execute if module runs as main program
if __name__ == '__main__' :
    
    # # create instance
    obj = UserVerify(input_path)
    data, idx_bounds = obj.main_func(file_id)
    
    if idx_bounds is not False and execute == 1:
        
        if idx_bounds.shape[0] == 0: # check for zero seizures
            obj.save_emptyidx(data.shape[0])
            
        else: # otherwise proceed with gui creation
    
            # get gui
            from verify_gui import matplotGui,fig,ax
            fig.suptitle('To Submit Press Enter; To Select Drag Mouse Pointer : '+file_id, fontsize=12)
               
            # init object
            callback = matplotGui(data,idx_bounds,obj, file_id)
            
            # add buttons
            axprev = plt.axes([0.625, 0.05, 0.13, 0.075]) # previous
            bprev = Button(axprev, 'Previous: <')
            bprev.on_clicked(callback.previous)
            axnext = plt.axes([0.765, 0.05, 0.13, 0.075]) # next
            bnext = Button(axnext, 'Next: >')
            bnext.on_clicked(callback.forward)
            axaccept = plt.axes([0.125, 0.05, 0.13, 0.075]) # accept
            baccept = Button(axaccept, 'Accept: y')
            baccept.on_clicked(callback.accept)
            axreject = plt.axes([0.265, 0.05, 0.13, 0.075]) # reject
            breject = Button(axreject, 'Reject: n')
            breject.on_clicked(callback.reject)
            axbox = plt.axes([0.5, 0.055, 0.05, 0.05]) # seizure number
            text_box = TextBox(axbox, 'Szr #', initial='0')
            text_box.on_submit(callback.submit)
            
            # add key press
            idx_out = fig.canvas.mpl_connect('key_press_event', callback.keypress)
            
            # set useblit True on gtkagg for enhanced performance
            span = SpanSelector(ax, callback.onselect, 'horizontal', useblit=True,
                rectprops=dict(alpha=0.5, facecolor='red'))
    
    

       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        