#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:28:28 2020

@author: -
"""

from keras.optimizers import SGD, Adam
import scipy.io as sio
import os
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization
from keras.models import load_model
from keras import models, Input, Model, layers
from keras import backend as K
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
import random
import keras.callbacks
import math
import glob
import time


from tensorflow.python.compiler.tensorrt import trt_convert as trt

########################################
# Set the CUDA visible device to GPU of choice, for example GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
########################################

########################################
# Set the parameters
SubIndLow = 0
SubIndHigh = 1130
iFold = 2
########################################

# Function to create a list of subjects from the mat file
def Create_SubList(MatSubList):
    SubList = []
    for iSub in range(len(MatSubList)):
        SubList.append(MatSubList[iSub][0][0])
    return SubList 

# Function to freeze the graph
def freeze_graph(model):
    graph = K.get_session().graph
    with graph.as_default():
        input_names = [inp.op.name for inp in model.inputs]
        output_names = [out.op.name for out in model.outputs]
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            K.get_session(), graph.as_graph_def(), output_names)
    return frozen_graph, input_names, output_names


# Load mask and other data as before ...
Mask = nib.load('/mnt/Data3/RfMRILab/Lubin/ContainerVolumes/Data/Reslice_GreyMask_02_91x109x91.img')
Mask = Mask.get_data()  

# Load subinfo
InputDir = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Data/'
SubDir = InputDir+'D46_SILCODE/Phenodata/Session1/SubID.mat'
MatSubList = sio.loadmat(SubDir)
MatSubList = MatSubList['SubID'] 
SubList = Create_SubList(MatSubList)
Dx = sio.loadmat(InputDir+'D46_SILCODE/Phenodata/Session1/Dx.mat')
Dx = Dx['Dx']
QC = sio.loadmat(InputDir+'D46_SILCODE/Phenodata/Session1/QC.mat')
QC = QC['QC']
        
SubList_Test = [id for id, qc in zip(SubList, QC) if qc != 0]   
SubList_Test = SubList_Test[SubIndLow:SubIndHigh]
Dx_Test = [id for id, qc in zip(Dx, QC) if qc != 0] 
Dx_Test = Dx_Test[SubIndLow:SubIndHigh] 
Data_Test = np.zeros(shape=(len(SubList_Test),96,120,86,2),dtype=np.float16)



# Load Data   
MetricList = ['wc1','mwc1']   
for iMetric in range(len(MetricList)):
    ImageDir = InputDir+'D46_SILCODE/MR_Results/VBM/'+MetricList[iMetric]+'/'
    for iSubject in range(len(SubList_Test)):
        FileDir = glob.glob(ImageDir+MetricList[iMetric]+'_D46_SILCODE_'+SubList_Test[iSubject]+'*.nii*')
        Data_3D = nib.load(FileDir[0]);
        Data_3D = np.nan_to_num(np.array(Data_3D.get_data()) * np.array(Mask))
        Data_Test[iSubject,:,:,:,iMetric] = Data_3D[12:108,13:133,16:102]


Data_Template = np.zeros(shape=(96,120,86,2),dtype=np.float16)
for iMetric in range(len(MetricList)):
    Data_Template_3D = nib.load('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/ReverseOccPredMaps_Template_ADNC5050_Pred0.499928/Template_'+MetricList[iMetric]+'_ADNC5050_Pred0.499928.nii');
    Data_Template_3D = np.nan_to_num(np.array(Data_Template_3D.get_data()) * np.array(Mask))
    Data_Template[:,:,:,iMetric] = Data_Template_3D[12:108,13:133,16:102]
Data_Template = Data_Template[np.newaxis, np.newaxis, np.newaxis, :, :, :]

    


# Load model
keras.backend.clear_session()
model = load_model('/mnt/Data3/RfMRILab/Lubin/ContainerVolumes/Code/ForYan/AD_Classifiers/20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold'+str(iFold)+'.h5')

# Freeze the graph and get input/output names
frozen_graph, input_names, output_names = freeze_graph(model)

# Configure TensorRT conversion parameters
converter = trt.TrtGraphConverter(
    input_graph_def=frozen_graph,
    nodes_blacklist=output_names,
    max_batch_size=64,
    max_workspace_size_bytes= 31*1024*1024*1024,
    precision_mode="FP16",
    minimum_segment_size=10, #16
    is_dynamic_op=True,
    maximum_cached_engines=1)
    
    
# Convert the graph using TRT
with tf.device('/gpu:0'):
    trt_graph = converter.convert()

# Run predictions sequentially on the GPU
output_node = tf.import_graph_def(trt_graph, name='', return_elements=output_names)  # Import the graph



# Set OCC and parallel parameters
Step = 8
OccSize = 12  #16
batch_size = 16  
Iter1 = math.ceil((Data_Test.shape[1]-OccSize)/Step)+1
Iter2 = math.ceil((Data_Test.shape[2]-OccSize)/Step)+1
Iter3 = math.ceil((Data_Test.shape[3]-OccSize)/Step)+1

#iSub = 0
for iSub in range(len(SubList_Test)):
    Data_Occ = np.tile(Data_Template,(Iter1,Iter2,Iter3,1,1,1,1))
    for i in range(Iter1):
        for j in range(Iter2):
            for k in range(Iter3):
                Data_Occ[i,j,k,i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86),:] \
                = Data_Test[iSub,i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86),:]
    
    Data_Occ = Data_Occ.reshape(-1, 96, 120, 86, 2)
    
    start_time = time.time()
    with tf.Session() as sess:
        # Get the input and output tensors
    #    for op in sess.graph.get_operations()[:10]:
    #        print(op.name)
        input_tensor = sess.graph.get_tensor_by_name('input_1_1:0')
        output_tensor = output_node[0].outputs[0]
        
        predictions_list = []
        num_batches = math.ceil(Data_Occ.shape[0] / batch_size)
        
        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, Data_Occ.shape[0])
            batch_samples = Data_Occ[start_index:end_index]
            
            # Run predictions on the current batch
            batch_predictions = sess.run(output_tensor, feed_dict={input_tensor: batch_samples})
            predictions_list.extend(batch_predictions.tolist())
            
            # Calculate and print progress
            progress_percentage = ((batch_index + 1) / num_batches) * 100
            print("Progress: subject{} - {:.2f}% completed.".format(iSub,progress_percentage))
        
        sub_predictions = sess.run(output_tensor, feed_dict={input_tensor: Data_Test[np.newaxis,iSub,:,:,:,:]})
        
        
    end_time = time.time()
    elapsed_time = end_time-start_time
    print(elapsed_time)
    
    # At this point, predictions_list contains the prediction results for all samples
    # If needed, you can convert the list back to a NumPy array
    OccPred = np.array(predictions_list)
    OccPred = OccPred.reshape(Iter1, Iter2, Iter3)
    
    PredMap = np.zeros(shape=(96,120,86))
    Count_Map = np.zeros(shape=(96,120,86))
    for i in range(Iter1):
        for j in range(Iter2):
            for k in range(Iter3):
                Shape = PredMap[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)].shape
                PredMap[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)] = \
                PredMap[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)] + np.ones(shape=Shape)*OccPred[i,j,k]
                Count_Map[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)] = \
                Count_Map[i*Step:min(i*Step+OccSize,96),j*Step:min(j*Step+OccSize,120),k*Step:min(k*Step+OccSize,86)] + np.ones(shape=Shape)
    
    
    PredMap = PredMap/Count_Map
    OutDir = '/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/ReverseOccPredMaps_TRT_ADNCTemplate_SILCODE_20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold'+str(iFold) \
    +'_Step'+str(Step)+'_OccSize'+str(OccSize)+'_'+str(SubIndLow+1)+'_'+str(SubIndHigh)+'/'
    os.makedirs(OutDir,exist_ok=True)
    DemoNii = nib.load('/mnt/Data4/RfMRILab/Lubin/ContainerVolumes/TransferLearningProject/Results/ReverseOcclusionMap_anatResolution_MIRIAD_20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold0.nii')
    Header = DemoNii.header 
    Affine = Header.get_best_affine()
    Brain = np.zeros(shape=(121,145,121))
    Brain[12:108,13:133,16:102] = PredMap
    BrainNii = nib.Nifti1Image(Brain,Affine,header=Header)
    nib.save(BrainNii,OutDir+'PredMap_Sub'+SubList_Test[iSub]+'_Dx'+str(np.squeeze(Dx_Test[iSub]))+'_Pred'+str(np.round(np.squeeze(sub_predictions),decimals=4))+'.nii')
    




