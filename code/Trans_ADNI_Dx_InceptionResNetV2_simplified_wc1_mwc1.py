#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:51:25 2019

@author: -Bin Lu, larslu@foxmail.com
"""

import sys
sys.path.append("./keras_applications")
from Inception_resnet_v2_3D import  conv3d_bn
from keras.optimizers import SGD, Adam
import scipy.io as sio
import os
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,AveragePooling3D, BatchNormalization
from keras.models import load_model
from keras import models,Input, Model, regularizers, layers
from keras import backend as K
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model,to_categorical
from keras.models import Sequential
from tensorflow.python.client import device_lib
import random
import keras.callbacks

def Create_SubList(MatSubList):
    SubList=[]
    for iSub in range(len(MatSubList)):
        Name = MatSubList[iSub][0][0]
        SubList.append(Name)
    return SubList  

def Shifting_Dataset(Data,Direction,Step):
    NewData = np.zeros(np.shape(Data),dtype=np.float16)
    if Direction == 'x':
        if Step > 0:
            NewData[:,Step:,:,:,:] = Data[:,:(np.shape(Data)[2]-Step),:,:,:]
        else:
            NewData[:,:(np.shape(Data)[1]-abs(Step)),:,:,:] = Data[:,abs(Step):,:,:,:]
    elif Direction == 'y':
        if Step > 0:
            NewData[:,:,Step:,:,:] = Data[:,:,:(np.shape(Data)[2]-Step),:,:]
        else:
            NewData[:,:,:(np.shape(Data)[2]-abs(Step)),:,:] = Data[:,:,abs(Step):,:,:]
    elif Direction == 'z':
        if Step > 0:
            NewData[:,:,:,:Step:,:] = Data[:,:,:,:(np.shape(Data)[2]-Step),:]
        else:
            NewData[:,:,:,:(np.shape(Data)[3]-abs(Step)),:] = Data[:,:,:,abs(Step):,:]        
    else:
        print('The Direction must be x, y or z!')
    return NewData

def Filp_Dataset(Data): # Flip brain (left/right brain)
    Index = list(range(np.shape(Data)[1]))
    Index.reverse()
    NewData = Data[:,Index,:,:,:]
    return NewData

nFold = 5
iFold = 0
loss = np.zeros(shape=(nFold))
acc = np.zeros(shape=(nFold))
loss1 = np.zeros(shape=(nFold))
acc1 = np.zeros(shape=(nFold))
DataName = 'AD'
Date = '20210106_0'
Type = 'Trans' # Trans FromScratch
Phase = 'Phase4'
nGPU = 4
batch_size = nGPU*6  
for iFold in range(nFold):
    PhenoInfo = sio.loadmat('./Phenotype_AD/AD_NC_Lfso_Fold'+str(iFold+1)+'.mat')
    SUbID = PhenoInfo['SUbID_Train']
    SubList_Train_Raw = Create_SubList(SUbID)
    Dx_Train_Raw = PhenoInfo['Dx_Train']
    SUbID = PhenoInfo['SUbID_Test']
    SubList_Test_Raw = Create_SubList(SUbID)
    Dx_Test_Raw = PhenoInfo['Dx_Test']
    QC_Corr_Binar_Train = PhenoInfo['QC_Corr_Binar_Train']
    QC_Corr_Binar_Test = PhenoInfo['QC_Corr_Binar_Test']
   
    SubList_Test = []
    Dx_Test = []
    Sex_Test = []
    SubList_Train = []
    Dx_Train = []
    Sex_Train = []
    for iSub in range(len(SubList_Train_Raw)):
        if  (Dx_Train_Raw[iSub]==0) & (QC_Corr_Binar_Train[iSub]>0):
            SubList_Train.append(SubList_Train_Raw[iSub]) 
            Dx_Train.append(0) 
        elif (Dx_Train_Raw[iSub]==1)& (QC_Corr_Binar_Train[iSub]>0):  
            SubList_Train.append(SubList_Train_Raw[iSub]) 
            Dx_Train.append(1)  
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
                 
    Data_Train_Raw = np.zeros(shape=(len(SubList_Train),96,120,86,2),dtype=np.float16)
    Data_Test = np.zeros(shape=(len(SubList_Test),96,120,86,2),dtype=np.float16)

    # Load Training Data  
    InputDir = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Data/D32_ADNI_1/MR_Results/VBM/' 
    MetricList = ['wc1','mwc1']   
    for iMetric in range(len(MetricList)):
        for iSub in range(len(SubList_Train)):
            try:
                Data_3D = nib.load(InputDir+MetricList[iMetric]+'/'+MetricList[iMetric]+'_D32_ADNI_1_'+SubList_Train[iSub]+'.nii.gz')
            except FileNotFoundError:
                Data_3D = nib.load(InputDir+MetricList[iMetric]+'/'+MetricList[iMetric]+'_D32_ADNI_1_'+SubList_Train[iSub]+'.nii')
            Data_3D = np.array(Data_3D.get_data()) * np.array(Mask)
            Data_Train_Raw[iSub,:,:,:,iMetric] = Data_3D[12:108,13:133,16:102]
            if iSub%500 == 0:
                    print('Have read training ',MetricList[iMetric],' ',iSub,'!')
                    
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
            if iSub%500 == 0:
                    print('Have read testing ',MetricList[iMetric],' ',iSub,'!')
                  
#    # Data augment
#    Data_Train_Fliped = Filp_Dataset(Data_Train_Raw);
#    Data_Train_Shifted = Shifting_Dataset(Data_Train_Raw,'y',2)
#    Data_Train = np.concatenate((Data_Train_Raw,Data_Train_Fliped,Data_Train_Shifted), axis=0)
#    Dx_Train = np.concatenate((Dx_Train_Raw,Dx_Train_Raw,Dx_Train_Raw), axis=0)
#    del Data_Train_Fliped
#    del Data_Train_Shifted
    
    # without data augment
    Data_Train = Data_Train_Raw

    index = [i for i in range(len(Data_Train))] 
    random.shuffle(index)
    Data_Train = Data_Train[index]
    Dx_Train = np.array(Dx_Train)
    Dx_Train = Dx_Train[index]
    
    index = [i for i in range(len(Data_Test))] 
    random.shuffle(index)
    Data_Test = Data_Test[index]
    Dx_Test = np.array(Dx_Test)
    Dx_Test = Dx_Test[index]
    
    keras.backend.clear_session()
    with tf.device('/cpu:0'):
        OldModel = load_model('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Model/20200427_Phase4_FromScratch_Gender_IncepResN_lr01_AllIn.h5')
        layer_names = [layer.name for layer in OldModel.layers]
        layer_idx = layer_names.index('StemEnd')
        x = keras.layers.MaxPooling3D(3, strides=2, name='mpool1')(OldModel.layers[layer_idx+1].output)
        x = conv3d_bn(x, 256, 3, padding='valid', name='conv1') #
        x = conv3d_bn(x, 256, 3, 2, name='conv2')
        x1 = keras.layers.MaxPooling3D(3, strides=2, name='mpool2')(x)
        x2 = conv3d_bn(x, 256, 3, 2, padding='valid', name='conv3')
        channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else 4
        x = keras.layers.Concatenate(axis=channel_axis, name='conc1')([x1,x2]) #nKernal = 1536
        x = keras.layers.GlobalAveragePooling3D(name='apool1')(x)
        x = Dropout(0.5,name='dropout1')(x)
        FinalOutput = Dense(1, activation='sigmoid')(x)
        model = Model(OldModel.input, FinalOutput)
        
    filepath = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Model/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'_AutoSave.h5'
    logpath = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/TensorBoard/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr0003_Lfso_Fold'+str(iFold)
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
                 keras.callbacks.TensorBoard(log_dir=logpath, update_freq='epoch')]
    nTest = (len(Dx_Test)//batch_size)*batch_size
    nTrain = (len(Dx_Train)//batch_size)*batch_size
    print('Old model loaded!')
    
#    set_trainable = False
#    for layer in model.layers:
#        if layer.name == 'conv_7b':
#            set_trainable = True
#        if set_trainable:
#            layer.trainable = True
#        else:
#            layer.trainable = False            
          
    sgd = SGD(lr=0.0003, decay=1e-3, momentum=0.9, nesterov=True)
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print('Training!')
    parallel_model.fit(Data_Train[5*batch_size:nTrain], Dx_Train[5*batch_size:nTrain], batch_size=batch_size, epochs=10, shuffle=True, initial_epoch=0, validation_data=(Data_Train[:5*batch_size],Dx_Train[:5*batch_size]),callbacks=callbacks) #,callbacks=callbacks, validation_data=(Data_Test[0:500],Dx_Test[0:500])
    
    # of note, if the sample size of test sample is not multiple of batch size, error occurs
    loss[iFold],acc[iFold] = parallel_model.evaluate(Data_Test[:nTest], Dx_Test[:nTest], batch_size=batch_size)    
    activations = parallel_model.predict(Data_Test[:nTest], batch_size=batch_size)
    np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'_Activation.npy',activations)
    np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'_Dx_Test.npy',Dx_Test)
    print('On this round, acc: '+str(acc[iFold])+', loss: '+str(loss[iFold]))
    model.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Model/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'.h5')
    
    # use the best model in training
    print('Test the best model!')
    keras.backend.clear_session()
    model = load_model('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Model/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'_AutoSave.h5')
    sgd = SGD(lr=0.0003, decay=1e-3, momentum=0.9, nesterov=True)
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    loss1[iFold],acc1[iFold] = parallel_model.evaluate(Data_Test[:nTest], Dx_Test[:nTest], batch_size=batch_size)    
    activations1 = parallel_model.predict(Data_Test[:nTest], batch_size=batch_size)
    print('On this round, acc1: '+str(acc1[iFold])+', loss1: '+str(loss1[iFold]))
    print('Have down fold No.'+str(iFold)+' !')
    np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'_Activation_Autosave.npy',activations1)
    
loss_mean = np.mean(loss)
acc_mean = np.mean(acc)
loss_mean1 = np.mean(loss1)
acc_mean1 = np.mean(acc1)
print('The average loss is '+str(loss_mean)+' !')
print('The average acc is '+str(acc_mean)+' !')
print('The average loss1 is '+str(loss_mean1)+' !')
print('The average acc1 is '+str(acc_mean1)+' !')




