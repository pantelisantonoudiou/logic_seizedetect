# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:02:29 2020

@author: Pante
"""

### imports ###
import os, time, tables, features, preprocess
import numpy as np

def get_data(main_path, exp_path, ch_num = [0,1] , inner_path=[], load_y = True):
    """
    get_data(main_path, exp_path, ch_num = [0,1] , inner_path=[], load_y = True)
    get_data(main_path, exp_path, ch_num = [0,1] , inner_path={'data_path':'filt_data'}, load_y = False)

    Parameters
    ----------
    main_path : Str
    exp_path : Str
    ch_num : List, channels to be loaded, optional, The default is [0,1].
    inner_path : Define inner folder for x or/and y data loading, optional The default is empty list.
    load_y : Bool, If False only data are returned, optional, The default is True.

    Returns
    -------
    data : ndarray (1d = segments, 2d = time, 3d = channels)
    y_data : 1D Int ndarray

    """
    
    if len(inner_path) == 0:
        inner_path = {'data_path':'reorganized_data', 'pred_path':'verified_predictions_pantelis'}
    
    # load lfp/eeg data
    f = tables.open_file(os.path.join(main_path, inner_path['data_path'], exp_path+'.h5'), mode = 'r') # open tables object
    data = f.root.data[:]; f.close() # load data
    data = data[:,:,ch_num] # get only desirer channels
    
    y_data = []
    if load_y == True: # get ground truth labels
        y_data = np.loadtxt(os.path.join(main_path,inner_path['pred_path'], exp_path+'.csv'), delimiter=',', skiprows=0)
        y_data = y_data.astype(np.bool) # conver to bool
        return data, y_data
    
    return data


def save_data(main_path, exp_path, data):
    """
    save_data(main_path, exp_path, data)

    Parameters
    ----------
    main_path : Str
    exp_path : Str
    data : ndarray

    Returns
    -------

    """
    
    try:
        # Saving Parameters
        atom = tables.Float64Atom() # declare data type 
        fsave = tables.open_file(os.path.join(main_path, exp_path+'.h5') , mode = 'w') # open tables object
        ds = fsave.create_earray(fsave.root, 'data', atom, # create data store 
                                    [0, data.shape[1], data.shape[2]])
        ds.append(data) # append data
        fsave.close() # close tables object
        return 1
    
    except:
        print('File could not be saved')
        return 0
    
# get one channel features metrics 
def get_features_single_ch(data,param_list):
    """
    get_features_single_ch(data,param_list)

    Parameters
    ----------
    data : 2D ndarray (1d = segments, 2d = time)
    param_list : ndarray with functions

    Returns
    -------
    x_data : 2D ndarray (rows = segments, columns = features)
    feature_labels : List
    """
    
    # Init data matrix
    x_data = np.zeros((data.shape[0], param_list.shape[0]))
    x_data = x_data.astype(np.double)
    
    feature_labels = [] # init labels list.
    for ii in range(param_list.shape[0]): # iterate over parameter list
    
        # append function name (feature) to list   
        feature_labels.append(param_list[ii].__name__)
        
        for i in range(data.shape[0]):    # iterate over segments
            x_data[i,ii] = param_list[ii](data[i,:]) # extract feature
            
    return x_data, feature_labels

# get cross-channel feature metrics
def get_features_crossch(data, param_list):
    """
    get_features_crossch(data, param_list)
    
    Parameters
    ----------
    data : 3D ndarray (1d = segments, 2d = time, 3d = channels)
    cross_ch_param_list : ndarray with functions

    Returns
    -------
    x_data : 2D ndarray (rows = segments, columns = features)
    feature_labels : List
    """
    
    # Init data matrix
    x_data = np.zeros((data.shape[0], param_list.shape[0]))
    x_data = x_data.astype(np.double)
    
    feature_labels = [] # init labels list
    for ii in range(param_list.shape[0]): # iterate over parameter list
    
        # append function name (feature) to list      
        feature_labels.append(param_list[ii].__name__)
        
        for i in range(data.shape[0]):    # iterate over segments
            x_data[i,ii] = param_list[ii](data[i,:,0], data[i,:,1]) # extract feature
        
    return x_data, feature_labels
    

# get features for all channels
def get_features_allch(data,param_list,cross_ch_param_list):
    """
    get_features_allch(data,param_list,cross_ch_param_list)

    Parameters
    ----------
    data : 3D ndarray (1d = segments, 2d = time, 3d = channels)
    param_list : ndarray with functions

    Returns
    -------
    x_data :  2D ndarray (rows = segments, columns = features)
    labels : np.array, feature names
    """
    labels = [] # make list to store labels
    x_data = np.zeros((data.shape[0],0),dtype=np.float) # make array to store all features
    
    # calculate single channel measures
    for ii in range(data.shape[2]):
        # get data and features labels per channel
        temp_data, feature_labels = get_features_single_ch(data[:,:,ii], np.array(param_list))
        x_data = np.concatenate((x_data, temp_data), axis=1) # append data
        
        str2append = '_' + str(ii) # get channel string
        labels += [s + str2append for s in feature_labels] # add to feature labels
    
    # calculcate cross-channel measures
    if len(cross_ch_param_list)>0:
        temp_data, feature_labels = get_features_crossch(data, np.array(cross_ch_param_list))
        x_data = np.concatenate((x_data, temp_data), axis=1) # append data
        labels += [s for s in feature_labels] # add to feature labels   
     
    return x_data, np.array(labels)


if __name__ == '__main__':
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    tic = time.time() # start timer
    # define single parameter list
    param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,
                  features.get_envelope_max_diff,)
    cross_ch_param_list = (features.cross_corr,)
    
    # Get data
    main_path = r'W:\Maguire Lab\Trina\2019\07-July\3514_3553_3639_3640'
    #r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data\3514_3553_3639_3640' # 3514_3553_3639_3640 3642_3641_3560_3514
    exp_path = '071919_3553b'#'071919_3553b'-seizures, 072519_3640a, 072719_3553a, 072819_3553a
    
    # Get data
    data, y_data = get_data(main_path, exp_path, ch_num = [0,1])
    
    # clean and filter data
    data = preprocess.preprocess_data(data,  clean = True, filt = True, verbose = 1)
    # data = data.reshape((int(data.shape[0]/2),int(data.shape[1]*2),2))
    
    # Get features to dataframe
    x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)

    # Normalize data
    x_data = StandardScaler().fit_transform(x_data)
    
    # get multiplied data and remap labels
    new_data = np.multiply(x_data[:,0:len(param_list)-1],x_data[:,len(param_list):x_data.shape[1]-1-len(cross_ch_param_list)])
    x_data = np.concatenate((new_data, x_data[:,x_data.shape[1]-1:]), axis=1)
    labels = [x.__name__ for x in param_list]; labels += [x.__name__ for x in cross_ch_param_list]
    
    # # Apply PCA for dimensionality reduction
    # pca = PCA()
    # projected = pca.fit_transform(x_data)
    
    # # plot contibutions
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    
    # # separate clusters
    # model = KMeans(n_clusters=2, random_state=0,n_init=50)
    # idx = model.fit_predict(x_data[:,1].reshape((-1,1)))
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # scatter = ax.scatter(projected[:,0], projected[:,1], c=y_data)
    
    fig = plt.figure()
    y = x_data[:,1]# x_data[:,6]*x_data[:,14]
    x =  np.linspace(1,y.shape[0],y.shape[0])
    
    threshold = np.mean(y) + 5*np.std(y)
    idx = y>threshold
    
    plt.plot(x, y) #,color = 'k')
    plt.scatter(x,y, c = y_data) # idx, y_data
    
    print('Elapse time =', time.time()-tic)

    ## get dataframe
    # df = pd.DataFrame(x_data, columns = labels)
    # df['target'] = y_data
    
    # from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
    # # Store predicted class labels of X
    # y_pred = x_data[:,1]>0.0004
    # y_pred = y_pred.astype(int)
    
    # # append metric
    # print('Accuracy :', accuracy_score(y_data, y_pred))
    # print('Sensitivity :', recall_score(y_data, y_pred))
    # print('Specificity :', precision_score(y_data, y_pred))
    
    # fpr, tpr, thresholds = roc_curve(y_data, y_pred) #roc curve
    # print('AUC :', auc(fpr, tpr))
       
















