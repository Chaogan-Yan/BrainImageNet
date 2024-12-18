#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 08:20:22 2020

@author: -
"""

import sys
sys.path.append("/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Code/keras-applications-master/keras_applications")
from Inception_resnet_v2_3D import InceptionResNetV2
import tensorflow as tf
from keras.optimizers import SGD, Adam
import scipy.io as sio
from keras.models import load_model
from keras import backend as K
import nibabel as nib
import numpy as np
from keras.utils import multi_gpu_model
import keras.callbacks 
import glob

def Create_SubList(MatSubList):
    SubList=[]
    for iSub in range(len(MatSubList)):
        SubList.append(MatSubList[iSub][0][0])
    return SubList  

InputDir = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Data/'

# Load mask
Mask = nib.load('/mnt/Data3/RfMRILab/Lubin/ContainerVolumes/Data/Reslice_GreyMask_02_91x109x91.img')
Mask = Mask.get_data()  

SubDir = InputDir+'D46_SILCODE/Phenodata/Session1/SubInfo.mat'
SubInfo = sio.loadmat(SubDir)
MatSubList = SubInfo['SubList'] 
SubList_Test = Create_SubList(MatSubList)

Data_all = np.zeros(shape=(len(SubList_Test),96,120,86,2),dtype=np.float16)

# Load Data   
MetricList = ['wc1','mwc1']   
for iMetric in range(len(MetricList)):
    ImageDir = InputDir+'D46_SILCODE/MR_Results/VBM/'+MetricList[iMetric]+'/'
    for iSubject in range(len(SubList_Test)):
        FileDir = glob.glob(ImageDir+MetricList[iMetric]+'_D46_SILCODE_'+SubList_Test[iSubject]+'*.nii*')
        Data_3D = nib.load(FileDir[0]);
        Data_3D = np.nan_to_num(np.array(Data_3D.get_data()) * np.array(Mask))
        Data_all[iSubject,:,:,:,iMetric] = Data_3D[12:108,13:133,16:102]

nFold = 5
batch_size = 2
nTest = (len(SubList_Test)//batch_size)*batch_size
Predictions = np.zeros(shape=(nTest,5))
for iFold in range(5): 
    keras.backend.clear_session()
    with tf.device('/cpu:0'):
        model = load_model('/mnt/Data3/RfMRILab/Lubin/ContainerVolumes/Code/ForYan/AD_Classifiers/20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'.h5')
    
    model.layers[-1].activation = keras.activations.linear
    sgd = SGD(lr=0.001, decay=2e-3, momentum=0.9, nesterov=True)
    
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print('New model generated! Ready to train!') 
    Predictions[:,iFold] = np.squeeze(parallel_model.predict(Data_all[:nTest],batch_size = batch_size))
    print('Have down fold '+str(iFold)) 
    
MeanPre = np.mean(Predictions,axis=1)

#np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/Prediction_LinearOutput_SILCODE_20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_wc1_mwc1.npy',MeanPre)
