# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:20:49 2020

@author: Pante
"""
import os, tables, features
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif,RFE
from sklearn.inspection import permutation_importance
from build_feature_data import preprocess_data, get_features_allch
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc


#2- add RFE - based on model, recursive feature elimination


main_path = r'C:\Users\Pante\Desktop\seizure_data_tb\Train_data'
ch_num = [0,1]
test_ratio = 0.2

# define parameter list
param_list = (features.autocorr, features.line_length, features.rms, features.mad, features.var, features.std, features.psd, features.energy,)
cross_ch_param_list = (features.cross_corr,)

# Get y data
f = tables.open_file(os.path.join(main_path, 'y_data.h5') , mode = 'r') # open tables object
y_data = f.root.data[:]; f.close()
y_data = y_data.astype(np.int)

# Get x data
f = tables.open_file(os.path.join(main_path, 'x_data.h5') , mode = 'r') # open tables object
data = f.root.data[:]; f.close()
data = data[:,:,ch_num] # get only desirer channels

# Clean and filter data
data = preprocess_data(data,  clean = True, filt = True)

# Get features and labels
x_data, labels = get_features_allch(data,param_list,cross_ch_param_list)
labels = np.array(labels) # convert to np array

# Normalize data
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# get multiplied data and remap labels
new_data = np.multiply(x_data[:,0:len(param_list)],x_data[:,len(param_list):x_data.shape[1]-len(cross_ch_param_list)])
x_data = np.concatenate((new_data, x_data[:,x_data.shape[1]-1:]), axis=1)
labels = [x.__name__ for x in param_list]; labels += [x.__name__ for x in cross_ch_param_list]
labels = np.array(labels)

# split training and testing data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = test_ratio, shuffle = True)


### -------- FEATURE SELECTION -----------###
# get dataframe
df = pd.DataFrame(x_train, columns = labels)
df['target'] = y_train
sns.heatmap(df.corr(), annot = True, cmap = 'viridis')

### select best features based on anova ###
fvalue_selector = SelectKBest(f_classif, k=2)
x_best = fvalue_selector.fit_transform(x_train, y_train)
best_labels = labels[fvalue_selector.get_support()]
f,p = f_classif(x_train,y_train)
plt.figure()
plt.plot(labels,f/np.max(f), label = 'ANOVA')

# Recursive feature elimination

# from sklearn.tree import DecisionTreeClassifier
# # Decision Tree
# rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=1)
# rfe.fit(x_train, y_train)
# plt.plot(labels, 1/rfe.ranking_, label = 'rfe decision tree' )

# Recursive feature elimination
from sklearn.linear_model import LogisticRegression
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=1)
rfe.fit(x_train, y_train)
plt.plot(labels, 1/rfe.ranking_, label = 'rfe logistic regression' )


# permutation
from sklearn.neighbors import KNeighborsClassifier
# define the model
model = KNeighborsClassifier()
# fit the model
model.fit(x_train, y_train)
# perform permutation importance
results = permutation_importance(model, x_train, y_train, scoring='accuracy')
# get importance
importance = results.importances_mean
plt.plot(labels, results.importances_mean, label = 'permutation k-neighbor' )


plt.legend()

### -------- UNSUPERVISED SECTION -----------###

# cluster
model = KMeans(n_clusters=2, random_state=0,n_init=50)
idx = model.fit_predict(x_best)

x = x_best[:,0]
y = x_best[:,1]

cdict = ['k','r']
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

for g in np.unique(y_train):
    ix = np.where(y_train == g)
    ax1.scatter(x[ix], y[ix], c = cdict[g], label = g, s = 8)
ax1.legend()
ax1.title.set_text('Real')
ax1.set_xlabel(best_labels[0])
ax1.set_ylabel(best_labels[1])
for g in np.unique(idx):
    ix = np.where(idx == g)
    ax2.scatter(x[ix], y[ix], c = cdict[g], label = g, s = 8)
ax2.legend()
ax2.title.set_text('Predicted')
ax2.set_xlabel(best_labels[0])
ax2.set_ylabel(best_labels[1])

# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
# y = x_train[:,6] #projected[:,0] #
# x =  np.linspace(1,y.shape[0],y.shape[0])

# for g in np.unique(idx):
#     ix = np.where(idx == g)
#     ax1.scatter(x[ix], y[ix], c = cdict[g], label = g, s = 8)
# ax1.legend()

# y_train = y_train.astype(np.int)
# for g in np.unique(y_train):
#     ix = np.where(y_train == g)
#     ax2.scatter(x[ix], y[ix], c = cdict[g], label = g, s = 8)
# ax2.legend()
# ###### ----------------- SUPERVISED SECTION ---------------#########

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Declare an of the KNN classifier class with the value with neighbors.
# model = KNeighborsClassifier(n_neighbors=2).fit(x_best, y_train)

# model = SVC(kernel='linear', C=10).fit(x_best, y_train)

# model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(x_train, y_train)
# plt.plot(labels,model.feature_importances_)
# model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(x_train, y_train)

# Store predicted class labels of X
# y_pred = model.predict(x_test)

## y_pred = x_test[:,1]>0.0004
## y_pred = y_pred.astype(int)

# # append metric
# print('Accuracy :', accuracy_score(y_test, y_pred))
# print('Sensitivity :', recall_score(y_test, y_pred))
# print('Specificity :', precision_score(y_test, y_pred))

# fpr, tpr, thresholds = roc_curve(y_test, y_pred) #roc curve
# print('AUC :', auc(fpr, tpr))
       



## plot mean, standard deviation ##

# feature_mean = np.zeros([2,len(labels)])
# feature_std= np.zeros([2,len(labels)])

# feature_mean[0,:] = np.mean(x_train[y_train==0],axis=0)
# feature_mean[1,:] = np.mean(x_train[y_train==1],axis=0)

# feature_std[0,:] = np.std(x_train[y_train==0],axis=0)
# feature_std[1,:] = np.std(x_train[y_train==1],axis=0)
# plt.figure()
# plt.errorbar(labels,feature_mean[1,:],yerr=feature_std[1,:], label ='seizure')
# plt.errorbar(labels,feature_mean[0,:],yerr=feature_std[0,:], label ='no seizure')
# plt.legend()





# @jit(nopython = True)
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

# #  remapped_array = remap_array(np.sort(feature)) # remap array from 0 to 100
# @jit(nopython = True)
# def remap_array(array):
#     min_value = array.min()
#     max_value = array.max()
#     a = (array - min_value) / (max_value - min_value) * 100
#     return a   
    














