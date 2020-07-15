#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 08:20:22 2020

@author: -Bin Lu, larslu@foxmail.com
"""

import sys
import tensorflow as tf
from keras.optimizers import SGD, Adam
import scipy.io as sio
from keras import layers, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,AveragePooling3D, BatchNormalization
from keras.models import load_model
from keras import backend as K
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
import keras.callbacks 
import glob
from sklearn.metrics import roc_curve, auc

K.set_floatx('float32') # supports float16, float32, float64

def Create_SubList(MatSubList):
    SubList=[]
    for iSub in range(len(MatSubList)):
        SubList.append(np.array2string(MatSubList[iSub][0][0])[1:-1])
    return SubList  

InputDir = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Data/'

# Load mask
Mask = nib.load('/mnt/Data3/RfMRILab/Lubin/ContainerVolumes/Data/Reslice_GreyMask_02_91x109x91.img')
Mask = Mask.get_data()  

SubDir = InputDir+'D38_OAS12/Phenodata/Session1/SubIDxiu.mat'
MatSubList = sio.loadmat(SubDir)
MatSubList = MatSubList['SUbID'] 
SubList = Create_SubList(MatSubList)
Dx = sio.loadmat(InputDir+'D38_OAS12/Phenodata/Session1/Dx.mat')
Dx = Dx['Dx']
QC = sio.loadmat(InputDir+'D38_OAS12/Phenodata/Session1/QC_Corr_Binar.mat')
QC = QC['QC_Corr_Binar']

### OASIS
SubList_Test = []
Dx_Test = []
n1 = 0
n2 = 0
for iSub in range(len(SubList)):
    if  (Dx[iSub]==0) & (QC[iSub]>0) :
        SubList_Test.append(SubList[iSub]) 
        Dx_Test.append(0) 
        n1 += 1
    elif (Dx[iSub]==1)& (QC[iSub]>0) :
        SubList_Test.append(SubList[iSub]) 
        Dx_Test.append(1) 
        n2 += 1

Data_all = np.zeros(shape=(len(SubList_Test),96,120,86,2),dtype=np.float16)

# Load Data   
MetricList = ['wc1','mwc1']   
for iMetric in range(len(MetricList)):
    ImageDir = InputDir+'D38_OAS12/MR_Results/VBM/'+MetricList[iMetric]+'/'
    for iSubject in range(len(SubList_Test)):
        FileDir = glob.glob(ImageDir+MetricList[iMetric]+'_'+SubList_Test[iSubject][0:-1]+'*.nii*')
        Data_3D = nib.load(FileDir[0]);
        Data_3D = np.nan_to_num(np.array(Data_3D.get_data()) * np.array(Mask))
        Data_all[iSubject,:,:,:,iMetric] = Data_3D[12:108,13:133,16:102]

### MCI
SubDir = InputDir+'D32_ADNI_1/Phenodata/Session1/MCI_All_1.mat'
SubInfo = sio.loadmat(SubDir)
MatSubList = SubInfo['MCI_SubID'] 
SubList_Test = Create_SubList(MatSubList)
Dx_Test1 = np.squeeze(SubInfo['MCI_Dx'])

Data_all1 = np.zeros(shape=(len(SubList_Test),96,120,86,2),dtype=np.float16)

# Load Data   
MetricList = ['wc1','mwc1']   
for iMetric in range(len(MetricList)):
    ImageDir = InputDir+'D32_ADNI_1/MR_Results/VBM/'+MetricList[iMetric]+'/'
    for iSubject in range(len(SubList_Test)):
        FileDir = glob.glob(ImageDir+MetricList[iMetric]+'_'+SubList_Test[iSubject][0:-1]+'*.nii*')
        Data_3D = nib.load(FileDir[0]);
        Data_3D = np.nan_to_num(np.array(Data_3D.get_data()) * np.array(Mask))
        Data_all1[iSubject,:,:,:,iMetric] = Data_3D[12:108,13:133,16:102]
        if iSubject%500 == 0:
            print('Have read testing ',MetricList[iMetric],' ',iSubject,'!')

nFold = 5
batch_size = 48
nTest = (len(Dx_Test)//batch_size)*batch_size
Predictions = np.zeros(shape=(nTest,nFold))

nTest1 = (len(Dx_Test1)//batch_size)*batch_size
Predictions1 = np.zeros(shape=(nTest1,nFold))
for iFold in range(nFold): 
    keras.backend.clear_session()
    with tf.device('/cpu:0'):
        model = load_model('./Classifier_AD/20200504_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'.h5')
        
    sgd = SGD(lr=0.001, decay=2e-3, momentum=0.9, nesterov=True)
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # predict OASIS
    Predictions[:,iFold] = np.squeeze(parallel_model.predict(Data_all[:nTest],batch_size = batch_size))
    print('Have down fold '+str(iFold)) 
    # predict MCI
    Predictions1[:,iFold] = np.squeeze(parallel_model.predict(Data_all1[:nTest1],batch_size = batch_size))
    print('Have down fold '+str(iFold)) 

MeanPre = np.mean(Predictions,axis=1)   
Pre_Binar = np.around(MeanPre,0)
acc_OASIS = np.squeeze((len(Pre_Binar)-sum(abs(Pre_Binar-Dx_Test[:nTest])))/len(Pre_Binar))
Dx_Test = np.array(Dx_Test)
TP = len(np.where(np.where(Pre_Binar==1,1,0) * np.where(Dx_Test[:nTest]==1,1,0))[0])
TN = len(np.where(np.where(Pre_Binar==0,1,0) * np.where(Dx_Test[:nTest]==0,1,0))[0])
FP = len(np.where(np.where(Pre_Binar==1,1,0) * np.where(Dx_Test[:nTest]==0,1,0))[0])
FN = len(np.where(np.where(Pre_Binar==0,1,0) * np.where(Dx_Test[:nTest]==1,1,0))[0])
Sensitivity_OASIS = TP/(TP+FN)
Specificity_OASIS = TN/(TN+FP)   
fpr_keras_OASIS, tpr_keras_OASIS, thresholds_keras_OASIS = roc_curve(Dx_Test[:nTest], Predictions[:,iFold])
auc_keras_OASIS = auc(fpr_keras_OASIS, tpr_keras_OASIS)
    
Threshold = 0.3
Index = MeanPre>Threshold
Pre_Binar1 = np.zeros(len(MeanPre))
Pre_Binar1[Index] = 1
acc1 = np.squeeze((len(MeanPre)-sum(abs(Pre_Binar1-Dx_Test[:nTest])))/len(MeanPre))
TP = len(np.where(np.where(Pre_Binar1==1,1,0) * np.where(Dx_Test[:nTest]==1,1,0))[0])
TN = len(np.where(np.where(Pre_Binar1==0,1,0) * np.where(Dx_Test[:nTest]==0,1,0))[0])
FP = len(np.where(np.where(Pre_Binar1==1,1,0) * np.where(Dx_Test[:nTest]==0,1,0))[0])
FN = len(np.where(np.where(Pre_Binar1==0,1,0) * np.where(Dx_Test[:nTest]==1,1,0))[0])
Sensitivity2 = TP/(TP+FN)
Specificity2 = TN/(TN+FP)

MeanPre = np.mean(Predictions1,axis=1)   
Pre_Binar = np.around(MeanPre,0)
acc_MCI = np.squeeze((len(Pre_Binar)-sum(abs(Pre_Binar-Dx_Test1[:nTest1])))/len(Pre_Binar))
Dx_Test1 = np.array(Dx_Test1)
TP = len(np.where(np.where(Pre_Binar==1,1,0) * np.where(Dx_Test1[:nTest1]==1,1,0))[0])
TN = len(np.where(np.where(Pre_Binar==0,1,0) * np.where(Dx_Test1[:nTest1]==0,1,0))[0])
FP = len(np.where(np.where(Pre_Binar==1,1,0) * np.where(Dx_Test1[:nTest1]==0,1,0))[0])
FN = len(np.where(np.where(Pre_Binar==0,1,0) * np.where(Dx_Test1[:nTest1]==1,1,0))[0])
Sensitivity_MCI = TP/(TP+FN)
Specificity_MCI = TN/(TN+FP)
fpr_keras_MCI, tpr_keras_MCI, thresholds_keras_MCI = roc_curve(Dx_Test1[:nTest1], Predictions1[:,iFold])
auc_keras_MCI = auc(fpr_keras_MCI, tpr_keras_MCI)
    
np.savez('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/Result_Test_on_OASIS_and_MCI_5FoldAverage.npz',
         acc_MCI = acc_MCI,
         Sensitivity_MCI = Sensitivity_MCI,
         Specificity_MCI = Specificity_MCI,
         auc_keras_MCI = auc_keras_MCI,
         acc_OASIS = acc_OASIS,
         Sensitivity_OASIS = Sensitivity_OASIS,
         Specificity_OASIS = Specificity_OASIS,
         auc_keras_OASIS = auc_keras_OASIS)


label = 'AD/NC (area = {auc}, acc = {acc})'
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_OASIS, tpr_keras_OASIS, label=label.format(auc=np.round(auc_keras_OASIS,3),acc=np.round(acc_OASIS,3)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

label = 'Conversion/Non (area = {auc}, acc = {acc})'
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_MCI, tpr_keras_MCI, label=label.format(auc=np.round(auc_keras_MCI,3),acc=np.round(acc_MCI,3)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

