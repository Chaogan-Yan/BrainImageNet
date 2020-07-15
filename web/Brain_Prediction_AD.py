from keras.optimizers import SGD, Adam
import scipy.io as sio
import os
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


#InputDir = '/Users/czk/Desktop/test/test-web/OnlineClassifier/DemoData/User02'
#OutputDir = '/Users/czk/Desktop/test/test-web/OnlineClassifier/DemoResult/User02'
def AD_Classifier(InputDir, OutputDir, DataType):
    FileList = os.listdir(InputDir)
    FileList1 = []
    for iFile in range(len(FileList)):
        Index = FileList[iFile].find('1')+1
        FileList1.append(FileList[iFile][Index:])
    FileListUni = list(set(FileList1))
    FileListUni.sort()
    Data_all = np.zeros(shape=(len(FileListUni),96,120,86,2),dtype=np.float16)
    Mask = nib.load('Reslice_BrainMask_05_91x109x91.img')
    Mask = Mask.get_data() 
    FullList = []
    ShortList = []
    for iSub in range(len(FileListUni)):
        try:
            wc1 = nib.load(InputDir+'/wc1'+FileListUni[iSub])
            mwc1 = nib.load(InputDir+'/mwc1'+FileListUni[iSub])
            FullList.append(iSub)
        except FileNotFoundError:
            ShortList.append(iSub)
        wc1 = np.array(wc1.get_data()) * np.array(Mask)
        mwc1 = np.array(mwc1.get_data()) * np.array(Mask)
        Data_all[iSub,:,:,:,0] = wc1[12:108,13:133,16:102]
        Data_all[iSub,:,:,:,1] = mwc1[12:108,13:133,16:102]
    with tf.device('/cpu:0'):
        Model0 = load_model('20200504_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold0.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model0.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction0 = Model0.predict(Data_all, batch_size=10)
        Model1 = load_model('20200504_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold1.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction1 = Model1.predict(Data_all, batch_size=10)
        Model2 = load_model('20200504_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold2.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model2.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction2 = Model2.predict(Data_all, batch_size=10)
        Model3 = load_model('20200504_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold3.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model3.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction3 = Model3.predict(Data_all, batch_size=10)
        Model4 = load_model('20200504_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold4.h5')
        sgd = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True) 
        Model4.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        Prediction4 = Model4.predict(Data_all, batch_size=10)
    Prediction=Prediction0*0.2+Prediction1*0.2+Prediction2*0.2+Prediction3*0.2+Prediction4*0.2
    file_write_obj = open(OutputDir+'/Prediction.txt', 'w')
    for i in range(len(FullList)):
        file_write_obj.write(FileListUni[FullList[i]]+'\\\t'+str(Prediction[i])+' \n')
    file_write_obj.close()
    file_write_obj = open(OutputDir+'/Prediction.txt', 'a')
    for i in range(len(ShortList)):
        file_write_obj.write(FileListUni[ShortList[i]]+'\tNaN\n ')
#    file_write_obj.write('Prediction Finished. If there is no result, please check the form of your input.')
    file_write_obj.close()