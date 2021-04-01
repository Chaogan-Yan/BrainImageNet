
# Predicting Alzheimers Disease (AD) status by Brain Image (preprocessed gray matter density/volume data) 
# Please preprocess your structural image to get the gray matter density data (wc1*) and gray matter volume data (mwc1*) by DPABI/DPARSF or SPM. The data required is NIfTI format (.nii or .nii.gz) in MNI space with a resolution of 91*109*91. 
# FORMAT python3 y_Predict_AD.py -i /in -o /out
# /in -- the input dir, should be:
#     1. The working dir of DPARSF (preprocessed by the DPARSF default parameter or DPARSF VBM parameter). There should be a 'T1ImgNewSegment' folder under this directory.
#     2. Alternatively, can be a directory of preprocessed gray matter density data (wc1*) and gray matter volume data (mwc1*). Under this directory, there were filenames as wc1_XXXXX.nii and mwc1_XXXXX.nii (XXXXX is subject ID).
# /out -- the output dir. There would be AD_Prediction.txt after prediction.
# ___________________________________________________________________________
# Written by YAN Chao-Gan 200710. Model credits also to Bin Lu and Zhi-Kai Chang.
# International Big-Data Center for Depression Research, Institute of Psychology, Chinese Academy of Sciences, Beijing, China
# ycg.yan@gmail.com


import os
import glob

from keras.optimizers import SGD, Adam
import scipy.io as sio
import tensorflow as tf
from keras import layers, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,AveragePooling3D, BatchNormalization
from keras.models import load_model, Sequential
from keras import models
from keras import backend as K
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
import keras.callbacks 


def AD_Classifier(InputDir, OutputDir):
    
    SegDir=glob.glob(InputDir+os.path.sep+'T1ImgNewSegment')

    if len(SegDir)>=1:
        SubList=[]
        wc1FileList=[]
        mwc1FileList=[]
        FileList = os.listdir(InputDir+os.path.sep+'T1ImgNewSegment')
        for iFile in range(len(FileList)):
            SubList.append(FileList[iFile])
            DirTemp=glob.glob(InputDir+os.path.sep+'T1ImgNewSegment'+os.path.sep+FileList[iFile]+os.path.sep+'wc1*')
            wc1FileList.append(DirTemp[0])
            DirTemp=glob.glob(InputDir+os.path.sep+'T1ImgNewSegment'+os.path.sep+FileList[iFile]+os.path.sep+'mwc1*')
            mwc1FileList.append(DirTemp[0])
    else:
        SubList=[]
        wc1FileList=[]
        mwc1FileList=[]
        DirTemp=glob.glob(InputDir+os.path.sep+'mwc1*')
        for iFile in range(len(DirTemp)):
            head_tail = os.path.split(DirTemp[iFile]) 
            SubList.append(head_tail[1][4:-4])
            wc1FileList.append(head_tail[0]+os.path.sep+head_tail[1][1:])
            mwc1FileList.append(DirTemp[iFile])

    Data_all = np.zeros(shape=(len(SubList),96,120,86,2),dtype=np.float16)
    Mask = nib.load('Reslice_BrainMask_05_91x109x91.img')
    Mask = Mask.get_data() 
    FullList = []
    ShortList = []
    for iSub in range(len(SubList)):
        try:
            wc1 = nib.load(wc1FileList[iSub])
            mwc1 = nib.load(mwc1FileList[iSub])
            FullList.append(iSub)
        except FileNotFoundError:
            ShortList.append(iSub)
        wc1 = np.array(wc1.get_data()) * np.array(Mask)
        mwc1 = np.array(mwc1.get_data()) * np.array(Mask)
        Data_all[iSub,:,:,:,0] = wc1[12:108,13:133,16:102]
        Data_all[iSub,:,:,:,1] = mwc1[12:108,13:133,16:102]
    with tf.device('/cpu:0'):
        Model0 = load_model('20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold0.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model0.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction0 = Model0.predict(Data_all, batch_size=10)
        Model1 = load_model('20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold1.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction1 = Model1.predict(Data_all, batch_size=10)
        Model2 = load_model('20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold2.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model2.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction2 = Model2.predict(Data_all, batch_size=10)
        Model3 = load_model('20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold3.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model3.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction3 = Model3.predict(Data_all, batch_size=10)
        Model4 = load_model('20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold4.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model4.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction4 = Model4.predict(Data_all, batch_size=10)
    Prediction=(Prediction0+Prediction1+Prediction2+Prediction3+Prediction4)/5
    
    file_write_obj = open(OutputDir+'/AD_Prediction.txt', 'w')
    for i in range(len(FullList)):
        file_write_obj.write(SubList[FullList[i]]+':\t'+str(Prediction[i])+' \n')
    file_write_obj.close()
    file_write_obj = open(OutputDir+'/AD_Prediction.txt', 'a')
    for i in range(len(ShortList)):
        file_write_obj.write(SubList[ShortList[i]]+':\tNaN\n ')#    file_write_obj.write('Prediction Finished. If there is no result, please check the form of your input.')
    file_write_obj.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict AD')
    parser.add_argument('-i', metavar='path', required=True,
                        help='the path to input dir')
    parser.add_argument('-o', metavar='path', required=True,
                        help='the path to output dir')
    args = parser.parse_args()
    AD_Classifier(InputDir=args.i, OutputDir=args.o)
