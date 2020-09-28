# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:22:20 2020

@author: Pante
"""


# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:18:28 2020

@author: Pante
"""


##    >>>>>>>>> USER INPUT <<<<<<<<          ##
# # Add path to raw data folder in following format -> r'PATH'
# input_path = r'C:\Users\Pante\Desktop\test1\raw_data'

# Add path to model
# model_path = r'models\cnn_train_ratio_1_ch_1and2_train03.h5'

               ## ---<<<<<<<< ##              
               
### ------------------------ IMPORTS -------------------------------------- ###               
import os,tables
import numpy as np
from keras.models import load_model
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from multich_dataPrep import lab2mat
from path_helper import sep_dir
from array_helper import find_szr_idx
from single_psd import single_psd
### ------------------------------------------------------------------------###
import matplotlib.pyplot as plt
import pdb
              
class modelPredict:
    """
    Class for batch seizure prediction
    """
    
    # class constructor (data retrieval)
    def __init__(self, input_path):
        """
        lab2mat(main_path)

        Parameters
        ----------
        input_path : STRING
            Raw data path.

        """
        # pass input path
        self.input_path = input_path

        # Get general and inner paths
        self.gen_path, innerpath = sep_dir(input_path,1)
        
        # load object properties as dict
        jsonfile = 'organized.json'
        obj_props = lab2mat.load(os.path.join(self.gen_path, jsonfile))
        self.org_rawpath = obj_props['org_rawpath']
        
        # create raw pred path
        rawpred_path = 'model_predictions'
        obj_props.update({'rawpred_path' : rawpred_path})
        self.rawpred_path = os.path.join(self.gen_path, rawpred_path)
        
        # get sampling rate
        self.fs = round(obj_props['fs'] / obj_props['down_factor'])
        self.win = obj_props['win']
        
        self.surround_time = 60 # time around seizures in seconds
        self.szr_pwr_thresh = 200 # szr power thresh         
        self.szr_segments =  np.array([2,2],dtype=int)  # segments before and after seizure segment


    def mainfunc(self, model_path):
        """
        mainfunc(input_path,model_path,ch_sel)
    
        Parameters
        ----------
        input_path : String, Path to raw data.

        model_path : String, Path to model.
        """
       
        # make path
        if os.path.exists( self.rawpred_path) is False:
            os.mkdir( self.rawpred_path)
        
        # get file list
        mainpath = os.path.join(self.gen_path, self.org_rawpath)
        verpath = os.path.join(self.gen_path, 'verified_predictions_pantelis')
        filelist = list(filter(lambda k: '.csv' in k, os.listdir(verpath)))
        
        # load model object to memory to get path
        model = load_model(model_path)
        
        
        # loop files (multilple channels per file)
        for i in tqdm(range(len(filelist))):
            
            # get organized data
            file_id = filelist[i].replace('.csv', '.h5')
            filepath = os.path.join(mainpath, file_id)
            f = tables.open_file(filepath, mode='r')
            data = f.root.data[:]
            data = data[:,:,0:2] # select channels
            # data = data.reshape(data.shape[0], data.shape[1], 1)
            
            # get predictions (1D-array)
            bin_pred = self.get_predictions(data, model)
   
            # refine predictions (1D-array)
            ref_pred = self.refine_seizures(data, bin_pred)

            # save predictions as .csv
            file_id = filelist[i].replace('.h5', '.csv')
            np.savetxt(os.path.join(self.rawpred_path,file_id), ref_pred, delimiter=',',fmt='%i')
            
            
    def power_thresh(self,data,idx):
        """
        power_thresh(self,data,idx)

        Parameters
        ----------
        data : 2D Numpy Array (rows = segments, columns = time bins)
            Raw data.
        idx : 2 column Numpy Array (rows = seizure segments)
            First column = seizure start segment.
            Second column = seizure end segment.

        Returns
        -------
        idx : Same as input index
            Only seizure segments that obay power threshold condition are kept.

        """
        
        # Set Parameters
        powerobj = single_psd(self.fs,0.5, self.win) # init PSD object
        freq_range = [2,40] # set range for power estimation
        outbins = round(self.surround_time/self.win) # convert bounds to bins
        logic_idx = np.zeros(idx.shape[0], dtype=int) # pre-allocate logic vector
        
        for i in range(idx.shape[0]): # iterate through seizure segments
            
            # get seizure and surrounding segments
            bef = data[idx[i,0] - outbins : idx[i,0] ,:].flatten() # segment before seizure
            after = data[idx[i,1] : idx[i,1] + outbins,:].flatten() # segment after seizure
            outszr = np.concatenate((bef,after),axis = None) # merge
            szr = data[idx[i,0] : idx[i,1],:].flatten() # seizure segment 
           
            # get power
            outszr_pwr = powerobj.powersd(outszr,freq_range) # around seizure
            szr_pwr = powerobj.powersd(szr,freq_range) # seizure
            
            # check if power difference meets threshold
            cond = (np.sum(np.mean(szr_pwr,axis = 1)) / np.sum(np.mean(outszr_pwr,axis = 1)))*100
            if cond > (self.szr_pwr_thresh):
                logic_idx[i] = 1
        
        # get index that passed threshold
        idx = idx[logic_idx == 1,:]
        return idx
            
            
    def get_predictions(self, data, model):
        """
        get_predictions(self, data, model)
    
        Parameters
        ----------
        data : Numpy array
    
        model : keras model
    
        ch_sel : List
            Containing selected channels.
    
        Returns
        -------
        binpred : Numpy array (rows = segments, columns = channels)
        Model binary predictions

        """
        # init array
        binpred = np.zeros(data.shape[0], dtype = int)
        
        # get scaler
        scaler = StandardScaler()
        
        # iterate over selected data channels
        for i in range(data.shape[2]):
            data[:,:,i] = scaler.fit_transform(data[:,:,i]) # normalize
            
        pred = model.predict(data) # get predictions
        # pdb.set_trace()
        # plt.figure()
        # plt.hist(pred[:,1],bins=1000)
        # binpred = np.argmax(pred, axis = 1) # index of largest score
        binpred = pred[:,1]>0.5
        binpred = binpred.astype(np.int)
        return binpred # return predictions
    
    
    def refine_seizures(self, data, binpred):
        """
        ref_pred = refine_seizures(self, data, binpred)

        Parameters
        ----------
        data : Numpy array
        binpred : 1D-numpy array, model predictions

        Returns
        -------
        ref_pred : Numpy array, refined predictions

        """
        
        # remove discontinous seizure segments
        idx_bounds = find_szr_idx(binpred, self.szr_segments)

        # Remove segments that do not exceed power threshold, based on ch1-vHPC
        idx_bounds = self.power_thresh(data[:,:,0],idx_bounds)
 
        # pre allocate file for refined prediction with zeros
        ref_pred = np.zeros(binpred.shape[0])
        
        # convert index to binary file
        for i in range(idx_bounds.shape[0]): # assign index to 1
            if idx_bounds[i,0] > 0:
                ref_pred[idx_bounds[i,0]:idx_bounds[i,1]+1] = 1
        
        return ref_pred
    

### add method    
    def get_feature_pred(self, file_id):
        """
        get_feature_pred(self, file_id)

        Parameters
        ----------
        file_id : Str

        Returns
        -------
        data : 3d Numpy Array (1D = segments, 2D = time, 3D = channel)
        bounds_pred : 2D Numpy Array (rows = seizures, cols = start and end points of detected seizures)

        """
        
        # Define parameter list
        param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
                      features.get_envelope_max_diff,) # single channel features
        cross_ch_param_list = (features.cross_corr, features.signal_covar, features.signal_abs_covar,) # cross channel features
        
        # Get data and true labels
        data = get_data(self.gen_path, file_id, ch_num = ch_list, inner_path={'data_path':'filt_data'}, load_y = False)
        
        # Extract features and normalize
        x_data, labels = get_features_allch(data,param_list, cross_ch_param_list) # Get features and labels
        x_data = StandardScaler().fit_transform(x_data) # Normalize data
        
        # Get predictions
        thresh = (np.mean(x_data) + self.thresh * np.std(x_data))   # get threshold vector
        y_pred_array = (x_data > thresh) # get predictions for all conditions
        y_pred = y_pred_array * self.weights * self.enabled    # get predictions based on weights and selected features
        y_pred = np.sum(y_pred, axis=1) / np.sum(self.weights * self.enabled) # normalize to weights and selected features
        y_pred = y_pred > 0.5                                       # get popular vote
        bounds_pred = find_szr_idx(y_pred, np.array([0,1]))         # get predicted seizure index
        
        # If seizures are detected proceed to refine them
        if bounds_pred.shape[0] > 0:
            
            # Merge seizures close together
            bounds_pred = merge_close(bounds_pred, merge_margin = 5)
            
            # Remove seizures where a feature (line length or power) is not higher than preceeding region
            idx = np.where(np.char.find(self.feature_names,'line_length_0')==0)[0][0]
            bounds_pred = self.refine_based_on_surround(x_data[:,idx], bounds_pred)    
        
        return bounds_pred        
# # Execute if module runs as main program
# if __name__ == '__main__':
    
#     # init object
#     obj = modelPredict(input_path)
    
#     # get predictions in binary format and store in csv
#     obj.mainfunc(model_path)    
    
    
    
   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            