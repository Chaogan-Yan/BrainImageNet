%% Subtyping MCI patients via individualized AD brain riskmaps
% When calculating, the "individualized AD brain riskmaps" are referred to as "Reverse occlusion prediction map" because the computational process is similar to the reverse of calculating an occlusion map at the individual level.

%% Calculate the mean individualized AD brain riskmaps for each participant
clc;clear;
nFold = 5; % 5 AD brain riskmaps were created for each participant based on the interpretable models derived from the AD classifiers pretrained in five-fold crossvalidation

InDir = '/mnt/Data4/RfMRILab/Lubin/DeepLearning/TransferLearningProject/Results';
OutDir = '/mnt/Data4/RfMRILab/Lubin/DeepLearning/TransferLearningProject/Results/ReverseOccPredMaps_TRT_ADNCTemplate_SILCODE_20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_5FoldMean_Step8_OccSize12_1_1130';
mkdir(OutDir);

DataAll = zeros(nFold,121,145,121,1129); % input size 121*145*121, sample size = 1129
for iFold = 1:nFold
    [Data, VoxelSize, FileList, Header] = y_ReadAll([InDir,filesep,...
        'ReverseOccPredMaps_TRT_ADNCTemplate_SILCODE_20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_Fold',...
        num2str(iFold-1),'_Step8_OccSize12_1_1130']); % load all the individualized AD brain riskmaps in each fold
    DataAll(iFold,:,:,:,:) = Data;
    disp(num2str(iFold))
end

DataMean = squeeze(mean(DataAll,1));
for iSub = 1:length(FileList)
    Ind1 = strfind(FileList{iSub},'PredMap_');
    Ind2 = strfind(FileList{iSub},'_Dx');
    y_Write(squeeze(DataMean(:,:,:,iSub)),Header,[OutDir,filesep,FileList{iSub}(Ind1:Ind2+3)]); % Write mean individualized AD brain riskmap for each participant
    disp(num2str(iSub))
end
    
%% Load phenotype information
clc;clear;
PhenotypeDir = '/mnt/Data4/RfMRILab/Lubin/Project/SILCODE_ReverseOCC/Phenotype/Phenotype_ZhangMingKai_20241021/';
load([PhenotypeDir,filesep,'baseline_dx.mat']);
load([PhenotypeDir,filesep,'Dx.mat']);
load([PhenotypeDir,filesep,'PredScore_TRT.mat']);
load([PhenotypeDir,filesep,'Progression.mat']);
load([PhenotypeDir,filesep,'QC.mat']);
load([PhenotypeDir,filesep,'SubID.mat']);

%% Load all individualized brain riskmaps for AD patients
InDir = '/mnt/Data4/RfMRILab/Lubin/DeepLearning/TransferLearningProject/Results/ReverseOccPredMaps_TRT_ADNCTemplate_SILCODE_20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_5FoldMean_Step8_OccSize12_1_1130';
OutDir = '/mnt/Data4/RfMRILab/Lubin/Project/SILCODE_ReverseOCC/Result/RiskMapClustering';
mkdir(OutDir);
ADIndex = find(baseline_dx==1 & QC);  % AD: baseline_dx==1
RiskMapAD = zeros(length(ADIndex),121,145,121);

for iSub = 1:length(ADIndex)
    temp = dir([InDir,filesep,'PredMap_Sub',ID{ADIndex(iSub)},'*']);
    [Data, VoxelSize, FileList, Header] = y_ReadAll([temp.folder,filesep,temp.name]);
    RiskMapAD(iSub,:,:,:) = Data;
    disp(num2str(iSub))
end
RiskMapAD = reshape(RiskMapAD,size(RiskMapAD,1),[]);

%% Apply grey matter mask 
[MaskData,MaskHeader] = y_Read('/mnt/Data3/RfMRILab/Lubin/DeepLearning/Data/Reslice_GreyMask_02_91x109x91.img'); % using a grey matter mask to exclude useless areas
MaskData1 = reshape(MaskData,[],1);
MaskIndex = find(MaskData1);
RiskMapAD = RiskMapAD(:,MaskIndex);

%% Determine the optimal number of clusters (k=3)
MaxK = 10;
sse = zeros(1,MaxK);

for k = 1:MaxK
    [~,~,sumd] = kmeans(RiskMapAD,k);
    sse(k) = sum(sumd);
    disp(num2str(k))
end

figure;
plot(1:MaxK,sse,'bx-');
save([OutDir,filesep,'SSE_Curve.mat'],'sse');

%% Do k-means clustering and PCA based on  individualized brain riskmaps of AD patients
nCluster = 3;
[idx,c] = kmeans(RiskMapAD,nCluster);
[coeff, score, ~, ~, explained] = pca(RiskMapAD);
X_pca = score(:, 1:2);

Cluster_ID = ID(ADIndex);
Cluster_Pred = PredScore_TRT(ADIndex);
Cluster_Subtype = idx;
Cluster_SubtypeCenter = c;
Cluster_PCA_X = X_pca;

save([OutDir,filesep,'ClusteringResults_AD'],'Cluster_ID','Cluster_Pred','Cluster_Subtype','Cluster_SubtypeCenter','Cluster_PCA_X');

%% Write cluster centers as NIFTI images
for iCluster = 1:nCluster
    Brain = zeros(size(MaskData1));
    Brain(MaskIndex) = c(iCluster,:);
    Brain = reshape(Brain,size(MaskData));
    y_Write(Brain, Header, [OutDir,filesep,'AD_ClusterCenter',num2str(iCluster)]);
end

%% Identify subtypes of MCI patients based on their spatial similarity to the three cluster centers 
clc;clear;
% Load phenotype information
load('/mnt/Data4/RfMRILab/Lubin/Project/SILCODE_ReverseOCC/Phenotype/Phenotype_ZhangMingKai_20241021/baseline_dx.mat');
load('/mnt/Data4/RfMRILab/Lubin/Project/SILCODE_ReverseOCC/Phenotype/Phenotype_ZhangMingKai_20241021/SubID.mat');

% Read the cluster centers 
[Data1,~] = y_Read('/mnt/Data4/RfMRILab/Lubin/Project/SILCODE_ReverseOCC/Result/RiskMapClustering/AD_ClusterCenter1.nii');
Data1 = reshape(Data1,1,[]);
[Data2,~] = y_Read('/mnt/Data4/RfMRILab/Lubin/Project/SILCODE_ReverseOCC/Result/RiskMapClustering/AD_ClusterCenter2.nii');
Data2 = reshape(Data2,1,[]);
[Data3,~] = y_Read('/mnt/Data4/RfMRILab/Lubin/Project/SILCODE_ReverseOCC/Result/RiskMapClustering/AD_ClusterCenter3.nii');
Data3 = reshape(Data3,1,[]);

% Apply grey matter mask
[MaskData,MaskHeader] = y_Read('/mnt/Data4/RfMRILab/Lubin/DeepLearning/TransferLearningProject/Template/Reslice_GreyMask_02_91x109x91.img');
MaskData1 = reshape(MaskData,[],1);
MaskIndex = find(MaskData1);

Template1 = Data1(MaskIndex);
Template2 = Data2(MaskIndex);
Template3 = Data3(MaskIndex);

% Load mean individualized brain riskmaps of MCI patients and calculate their spatial similarity to the three cluster centers 
InDir = '/mnt/Data4/RfMRILab/Lubin/DeepLearning/TransferLearningProject/Results/ReverseOccPredMaps_TRT_ADNCTemplate_SILCODE_20210106_0_Phase4_Trans_AD_IncepResN_lr0003_Lfso_5FoldMean_Step8_OccSize12_1_1130';
ID_MCI = ID(find(baseline_dx==3));
CorrList = [];
for iSub = 1:length(ID_MCI)
    temp = dir([InDir,filesep,'PredMap_Sub',ID_MCI{iSub},'*.nii']);
    [Map,~] = y_Read([temp.folder,filesep,temp.name]);
    Map = reshape(Map,1,[]);
    Map_Masked = Map(MaskIndex);
    Corr1 = corrcoef(Map_Masked,Template1);
    Corr2 = corrcoef(Map_Masked,Template2);
    Corr3 = corrcoef(Map_Masked,Template3);
    CorrList(iSub,:) = [Corr1(1,2),Corr2(1,2),Corr3(1,2)];
    disp(num2str(iSub));
end

% In subsequent analyses, patients are assigned to a subtype based on which cluster center their riskmap most closely resembles. If a patient's riskmap is too similar to two subtypes, they are not assigned a specific subtype. 
save('/mnt/Data4/RfMRILab/Lubin/Project/SILCODE_ReverseOCC/Result/RiskMapClustering/Baseline_MCI_Subtype.mat','ID_MCI','CorrList');
