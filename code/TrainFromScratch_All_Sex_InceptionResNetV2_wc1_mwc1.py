#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:51:25 2019

@author: -Bin Lu, larslu@foxmail.com
"""

import sys
sys.path.append("./keras_applications")
from Inception_resnet_v2_3D import InceptionResNetV2
from keras.optimizers import SGD, Adam
import scipy.io as sio
import os
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,AveragePooling3D, BatchNormalization
from keras.models import load_model
from keras import models,Input, Model, layers
import nibabel as nib
import numpy as np
from keras.utils import multi_gpu_model
import random
import keras.callbacks
import glob
import threading
import gc

def Create_SubList(MatSubList):
    SubList=[]
    for iSub in range(len(MatSubList)):
        Name = MatSubList[iSub][0][0]
        SubList.append(Name)
    return SubList  

def Shifting_Dataset(Data,Direction,Step): # For data augment, shift brains
    NewData = np.zeros(np.shape(Data))
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

def Filp_Dataset(Data): # # For data augment, flip brains (left/right brain)
    Index = list(range(np.shape(Data)[1]))
    Index.reverse()
    NewData = Data[:,Index,:,:,:]
    return NewData

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
loss = np.zeros(shape=(nFold))
acc = np.zeros(shape=(nFold))
loss1 = np.zeros(shape=(nFold))
acc1 = np.zeros(shape=(nFold))
DataName = 'Gender'
Date = '20200420'
Type = 'FromScratch' # Trans
Phase = 'Phase4'
nGPU = 4
batch_size = nGPU*6

for iFold in range(nFold):
    PhenoInfo = sio.loadmat('./Phenotype_Sex/Lfso_Fold'+str(iFold+1)+'.mat')
    
    SUbID = PhenoInfo['SUbID_Train']
    SubList_Train = Create_SubList(SUbID)
    Sex_Train = PhenoInfo['Sex_Train']
    # These QC score were calculated and binarized accoding to the correlation between individual images and the grand mean image, 0 or 1
    QC_Corr_Train = PhenoInfo['QC_Corr_Binar_Train'] 
    # These QC score were rated by visual check, range from 1 to 5
    QC_Rate_Train = PhenoInfo['QC_Train']
    Dir_Train = PhenoInfo['Dir_Train']
    
    SUbID = PhenoInfo['SUbID_Test']
    SubList_Test = Create_SubList(SUbID)
    Sex_Test = PhenoInfo['Sex_Test']
    QC_Corr_Test = PhenoInfo['QC_Corr_Binar_Test']
    QC_Rate_Test = PhenoInfo['QC_Test']
    Dir_Test = PhenoInfo['Dir_Test']
    
    QC_2Way_Train = np.array(QC_Corr_Train) * np.array(QC_Rate_Train) 
    QCGood_Train = np.array(np.where(QC_2Way_Train > 2))
    SubList_Train = np.array(SubList_Train)[QCGood_Train[0,:]]
    Sex_Train = np.array(Sex_Train)[QCGood_Train[0,:]]
    Dir_Train = np.array(Dir_Train)[QCGood_Train[0,:]]
    
    QC_2Way_Test = np.array(QC_Corr_Test) * np.array(QC_Rate_Test) 
    QCGood_Test = np.array(np.where(QC_2Way_Test > 2))    
    SubList_Test = np.array(SubList_Test)[QCGood_Test[0,:]]
    Sex_Test = np.array(Sex_Test)[QCGood_Test[0,:]]
    Dir_Test = np.array(Dir_Test)[QCGood_Test[0,:]]
    
    index = [i for i in range(len(SubList_Train))] 
    random.shuffle(index)
    SubList_Train = SubList_Train[index]
    Sex_Train = np.nan_to_num(Sex_Train[index])
    Dir_Train = Dir_Train[index]
    
    index = [i for i in range(len(SubList_Test))] 
    random.shuffle(index)
    SubList_Test = SubList_Test[index]
    Sex_Test = np.nan_to_num(Sex_Test[index])
    Dir_Test = Dir_Test[index]
    
    # Load mask
    Mask = nib.load('/mnt/Data3/RfMRILab/Lubin/ContainerVolumes/Data/Reslice_GreyMask_02_91x109x91.img')
    Mask = Mask.get_data()  
                 
    Data_Train = np.zeros(shape=(len(SubList_Train),96,120,86,2),dtype=np.float16)
    Data_Test = np.zeros(shape=(len(SubList_Test),96,120,86,2),dtype=np.float16)
    MetricList = ['wc1','mwc1'] 

    # Load Training Data
    for iMetric in range(len(MetricList)):
        ParaReadImg(100,MetricList,iMetric,SubList_Train,Dir_Train,Mask,Data_Train)
        
    # Load Testing Data  
    for iMetric in range(len(MetricList)):
        ParaReadImg(100,MetricList,iMetric,SubList_Test,Dir_Test,Mask,Data_Test)
    
    keras.backend.clear_session()
    with tf.device('/cpu:0'):
        OldModel = InceptionResNetV2(include_top=True, weights=None, input_shape = (96, 120, 86, 2))#, input_tensor=RawInput
        OldModel.layers.pop()   
        x = Dropout(0.5)(OldModel.layers[-1].output)
        FinalOutput = Dense(1, activation='sigmoid')(x)
        model = Model(OldModel.input, FinalOutput)
      
    sgd = SGD(lr=0.01, decay=3e-3, momentum=0.9, nesterov=True)
    parallel_model = multi_gpu_model(model, gpus=nGPU)
    #'sparse_categorical_crossentropy'
    parallel_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print('New model generated! Ready to train!')

    filepath = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Model/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr01_Lfso_Fold'+str(iFold)+'_AutoSave.h5'
    logpath = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/TensorBoard/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr01_Lfso_Fold'+str(iFold)
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
                 keras.callbacks.TensorBoard(log_dir=logpath, update_freq=batch_size*50)]
    nTest = (len(Sex_Test)//batch_size)*batch_size
    nTrain = (len(Sex_Train)//batch_size)*batch_size
    parallel_model.fit(Data_Train[25*batch_size:nTrain], Sex_Train[25*batch_size:nTrain], batch_size=batch_size, epochs=10, shuffle=True, initial_epoch=0, validation_data=(Data_Train[:25*batch_size],Sex_Train[:25*batch_size]),callbacks=callbacks) #,callbacks=callbacks, validation_data=(Data_Test[0:500],Sex_Test[0:500])
    # of note, if the sample size of test sample is not multiple of batch size, error occurs
    loss[iFold],acc[iFold] = parallel_model.evaluate(Data_Test[:nTest], Sex_Test[:nTest], batch_size=batch_size)    
    activations = parallel_model.predict(Data_Test[:nTest], batch_size=batch_size)
    np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr01_Lfso_Fold'+str(iFold)+'_Activation.npy',activations)
    np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr01_Lfso_Fold'+str(iFold)+'_Sex_Test.npy',Sex_Test)
    print('On this round, acc: '+str(acc[iFold])+', loss: '+str(loss[iFold]))
    model.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Model/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr01_Lfso_Fold'+str(iFold)+'.h5')
    
    # use the best model in training
    print('Test the best model!')
    keras.backend.clear_session()
    model = load_model('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Model/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr01_Lfso_Fold'+str(iFold)+'_AutoSave.h5')
    sgd = SGD(lr=0.0003, decay=1e-3, momentum=0.9, nesterov=True)
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    loss1[iFold],acc1[iFold] = parallel_model.evaluate(Data_Test[:nTest], Sex_Test[:nTest], batch_size=batch_size)    
    activations1 = parallel_model.predict(Data_Test[:nTest], batch_size=batch_size)
    print('On this round, acc1: '+str(acc1[iFold])+', loss1: '+str(loss1[iFold]))
    print('Have down fold No.'+str(iFold)+' !')
    np.save('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/'+Date+'_'+Phase+'_'+Type+'_'+DataName+'_IncepResN_lr01_Lfso_Fold'+str(iFold)+'_Activation_Autosave.npy',activations1)
    
    # clear memory
    del Data_Test, Data_Train
    gc.collect()
    
loss_mean = np.mean(loss)
acc_mean = np.mean(acc)
loss_mean1 = np.mean(loss1)
acc_mean1 = np.mean(acc1)
print('The average loss is '+str(loss_mean)+' !')
print('The average acc is '+str(acc_mean)+' !')
print('The average loss1 is '+str(loss_mean1)+' !')
print('The average acc1 is '+str(acc_mean1)+' !')




