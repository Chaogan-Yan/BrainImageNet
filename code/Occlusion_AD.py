#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:28:28 2020

@author: -Bin Lu, larslu@foxmail.com
"""


from keras.optimizers import SGD, Adam
import scipy.io as sio
import os
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,AveragePooling3D, BatchNormalization
from keras.models import load_model
from keras import models,Input, Model, layers
from keras import backend as K
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from keras.models import Sequential
from tensorflow.python.client import device_lib
import random
import keras.callbacks
import math
import glob
import threading

def Create_SubList(MatSubList):
    SubList=[]
    for iSub in range(len(MatSubList)):
        Name = MatSubList[iSub][0][0]
        SubList.append(Name)
    return SubList  

def ReadImg(MetricList,iMetric,SubList,iSubject,Dir,Mask,Data_all):
    InDir = Dir[iSubject][0][0]
    InDir = InDir.replace('DeepLearning','ContainerVolumes')
    FileDir = glob.glob(InDir+'/MR_Results/VBM/'+MetricList[iMetric]+'/'+MetricList[iMetric]+'_'+SubList[iSubject]+'*.nii*')
    Data_3D = nib.load(FileDir[0]);
    Data_3D = np.nan_to_num(np.array(Data_3D.get_data()) * np.array(Mask))
    Data_all[int(iSubject),:,:,:,iMetric] = Data_3D[12:108,13:133,16:102]
    del Data_3D
    if iSubject%500 == 0:
        print('Have read ',MetricList[iMetric],' ',str(iSubject),'!')
    return    
    
def ParaReadImg(nThread,MetricList,iMetric,SubList,Dir,Mask,Data_all):
    if len(SubList)%nThread==0:
        nPart = len(SubList)//nThread
    else:
        nPart = len(SubList)//nThread+1
    for iPart in range(nPart):
        Low = iPart*nThread
        High = min((iPart+1)*nThread,len(SubList))
        threads = []
        for iSubject in range(Low,High):
            t = threading.Thread(target = ReadImg, args=(MetricList,iMetric,SubList,iSubject,Dir,Mask,Data_all))
            threads.append(t)
            t.start()
            t.join()
    return

InputDir = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Data/'
DatasetList = os.listdir(InputDir)

nFold = 5
iFold = 0
Step = 6
OccSize = 12
DataName = 'AD'
Date = '20210106_0'
Type = 'Trans' # Trans
nGPU = 4
batch_size = nGPU*12
for iFold in range(iFold,iFold+1):
    PhenoInfo = sio.loadmat('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Data/D32_ADNI_1/Phenodata/Session1/AD_NC_Lfso_Fold'+str(iFold+1)+'.mat')
    SUbID = PhenoInfo['SUbID_Test']
    SubList_Test_Raw = Create_SubList(SUbID)
    Dx_Test_Raw = PhenoInfo['Dx_Test']
    QC_Corr_Binar_Test = PhenoInfo['QC_Corr_Binar_Test']
   
    SubList_Test = []
    Dx_Test = []
    Sex_Test = []
 
    for iSub in range(len(SubList_Test_Raw)):
        if  (Dx_Test_Raw[iSub]==0) & (QC_Corr_Binar_Test[iSub]>0):
            SubList_Test.append(SubList_Test_Raw[iSub]) 
            Dx_Test.append(0) 
        elif (Dx_Test_Raw[iSub]==1)& (QC_Corr_Binar_Test[iSub]>0):
            SubList_Test.append(SubList_Test_Raw[iSub]) 
            Dx_Test.append(1) 

            
    # Load mask
    Mask = nib.load('/mnt/Data3/RfMRILab/Lubin/ContainerVolumes/Data/Reslice_GreyMask_02_91x109x91.img')
    Mask = Mask.get_data()  
                 
    Data_Test = np.zeros(shape=(len(SubList_Test),96,120,86,2))
                  
     # Load Testing Data  
    InputDir = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Data/D32_ADNI_1/MR_Results/VBM/' 
    MetricList = ['wc1','mwc1']   
    for iMetric in range(len(MetricList)):
        for iSub in range(len(SubList_Test)):
            try:
                Data_3D = nib.load(InputDir+MetricList[iMetric]+'/'+MetricList[iMetric]+'_D32_ADNI_1_'+SubList_Test[iSub]+'.nii.gz')
            except FileNotFoundError:
                Data_3D = nib.load(InputDir+MetricList[iMetric]+'/'+MetricList[iMetric]+'_D32_ADNI_1_'+SubList_Test[iSub]+'.nii')
            Data_3D = np.array(Data_3D.get_data()) * np.array(Mask)
            Data_Test[iSub,:,:,:,iMetric] = Data_3D[12:108,13:133,16:102]
            if iSub%100 == 0:
                print('Have read testing ',MetricList[iMetric],' ',iSub,'!')
        
    Iter1 = math.ceil((Data_Test.shape[1]-OccSize)/Step)+1
    Iter2 = math.ceil((Data_Test.shape[2]-OccSize)/Step)+1
    Iter3 = math.ceil((Data_Test.shape[3]-OccSize)/Step)+1
    
    OccMap = np.zeros(shape=(Iter1,Iter2,Iter3))
    
    keras.backend.clear_session()
    with tf.device('/cpu:0'):
        model = load_model('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Model/20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'.h5')
    
    sgd = SGD(lr=0.0003, decay=2e-3, momentum=0.9, nesterov=True)
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # of note, if the sample size of test sample is not multiple of batch size, error occurs
    nTest = (len(Dx_Test)//batch_size)*batch_size
    n = 0
    n0 = Iter1*Iter2*Iter3
    
    loss0,acc0 = parallel_model.evaluate(Data_Test[:nTest], Dx_Test[:nTest], batch_size=batch_size) 
    for i in range(Iter1):
        for j in range(Iter2):
            for k in range(Iter3):
                Data_Occ = np.copy(Data_Test)
                Data_Occ[:,i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86),:] = 0
                loss,acc = parallel_model.evaluate(Data_Occ[:nTest], Dx_Test[:nTest], batch_size=batch_size)    
                OccMap[i,j,k] = acc0-acc # for fold5
#                OccMap[i,j,k] = 1-acc # for fold5
                n += 1
                print('Have down i-',str(i),' j-',str(j),' k-',str(k),', acc = ',str(np.round(acc*100,2)),'%.')
                print('Have down ',str(np.round(n/n0*100,1)), '% of occlusion map!')
    np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/OcclusionMap_'+Date+'_'+Type+'_'+DataName+'_Fold'+str(iFold)+'.npy',OccMap)
    
    OccMap_anat = np.zeros(shape=(96,120,86))
    Count_Map = np.zeros(shape=(96,120,86))
    n = 0
    for i in range(Iter1):
        for j in range(Iter2):
            for k in range(Iter3):
                Shape = OccMap_anat[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)].shape
                OccMap_anat[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)] = \
                OccMap_anat[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)] + OccMap[i,j,k]*np.ones(shape=Shape)
                Count_Map[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)] = \
                Count_Map[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)] + np.ones(shape=Shape)
                n += 1
                print('Have down ',str(np.round(n/n0*100,1)), '% of occlusion map!')
    OccMap_anat = OccMap_anat/Count_Map
    np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/OcclusionMap_anat_'+Date+'_'+Type+'_'+DataName+'_Fold'+str(iFold)+'.npy',OccMap_anat)
               
    

    

    
    
    
    
    
    
    
    
    
    
    