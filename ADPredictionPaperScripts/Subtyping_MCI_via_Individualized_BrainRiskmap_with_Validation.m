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


%% Additional cluster validation requested by reviewer
% IMPORTANT:
% The original k = 3 clustering result above (idx and c) is retained as the
% MAIN RESULT. The analyses in this section are validation analyses only.
% They do NOT replace idx, c, Cluster_Subtype, or Cluster_SubtypeCenter and
% therefore do not change the subtype definitions used in the manuscript.
%
% Validation outputs:
%   1) Number of AD participants in each ORIGINAL cluster
%   2) Silhouette width of the ORIGINAL k = 3 clustering solution
%   3) Mean silhouette width for candidate k = 2:10 (supporting analysis)
%   5) Stability across 100 repeated k-means++ initializations at k = 3.
%      The run with the lowest SSE is identified, and stability is quantified
%      by ARI relative to that best-SSE repeated-initialization solution

ValidationDir = [OutDir,filesep,'ClusterValidation'];
mkdir(ValidationDir);

%% V1. Cluster sizes and silhouette width of the ORIGINAL main solution
load([OutDir,filesep,'ClusteringResults_AD']);
nCluster = 3;
Original_idx = Cluster_Subtype;      % preserve original main clustering labels
Original_c   = Cluster_SubtypeCenter;        % preserve original main cluster centers

Original_ClusterSize = accumarray(Original_idx,1,[nCluster,1]);
Original_Silhouette = silhouette(RiskMapAD,Original_idx,'sqeuclidean');
Original_MeanSilhouette = mean(Original_Silhouette,'omitnan');
Original_MedianSilhouette = median(Original_Silhouette,'omitnan');

fprintf('\n============================================================\n');
fprintf('VALIDATION OF ORIGINAL k = %d CLUSTERING SOLUTION\n',nCluster);
fprintf('============================================================\n');
for iCluster = 1:nCluster
    fprintf('Original Cluster %d: n = %d\n',iCluster,Original_ClusterSize(iCluster));
end
fprintf('Original mean silhouette width = %.4f\n',Original_MeanSilhouette);
fprintf('Original median silhouette width = %.4f\n',Original_MedianSilhouette);

figure('Color','w');
silhouette(RiskMapAD,Original_idx,'sqeuclidean');
title(sprintf('Silhouette plot of original clustering (k = %d)',nCluster));
exportgraphics(gcf,[ValidationDir,filesep,'Original_k3_SilhouettePlot.png'],'Resolution',300);
savefig(gcf,[ValidationDir,filesep,'Original_k3_SilhouettePlot.fig']);

%% V2. Mean silhouette width across candidate k = 2:10
% This is an additional internal-validity analysis only. It is NOT used to
% redefine the original k = 3 solution.
Validation_MaxK = 10;
Validation_KList_Sil = 2:Validation_MaxK;
Validation_MeanSilhouetteByK = nan(size(Validation_KList_Sil));
Validation_SSEByK = nan(1,Validation_MaxK);
Validation_nRepCriterion = 20;
Validation_RandomSeed = 20260820;

rng(Validation_RandomSeed,'twister');
for ii = 1:length(Validation_KList_Sil)
    k_val = Validation_KList_Sil(ii);
    [idx_val,~,sumd_val] = kmeans(RiskMapAD,k_val, ...
        'Distance','sqeuclidean', ...
        'Start','plus', ...
        'Replicates',Validation_nRepCriterion, ...
        'MaxIter',1000, ...
        'Display','off');

    sil_val = silhouette(RiskMapAD,idx_val,'sqeuclidean');
    Validation_MeanSilhouetteByK(ii) = mean(sil_val,'omitnan');
    Validation_SSEByK(k_val) = sum(sumd_val);

    fprintf('Validation k = %d: mean silhouette = %.4f\n', ...
        k_val,Validation_MeanSilhouetteByK(ii));
end

figure('Color','w');
plot(Validation_KList_Sil,Validation_MeanSilhouetteByK,'o-','LineWidth',1.5);
xlabel('Number of clusters (k)');
ylabel('Mean silhouette width');
title('Silhouette validation across candidate k');
grid on;
exportgraphics(gcf,[ValidationDir,filesep,'SilhouetteAcrossK.png'],'Resolution',300);
savefig(gcf,[ValidationDir,filesep,'SilhouetteAcrossK.fig']);
save([ValidationDir,filesep,'MeanSilhouetteAcrossK'],'Validation_MeanSilhouetteByK');


%% V3. Stability across repeated k-means++ initializations
% Run k-means 100 times with independent k-means++ initializations at k = 3.
% The run with the lowest SSE is identified as the best repeated-initialization
% solution. Stability is then quantified by the ARI between each repeated run
% and this best-SSE solution. Importantly, this validation does NOT replace
% the ORIGINAL clustering used for the main results.
Validation_nStabilityRuns = 100;
Validation_idxAll = zeros(size(RiskMapAD,1),Validation_nStabilityRuns);
Validation_SSE = nan(Validation_nStabilityRuns,1);

fprintf('\nRunning %d repeated k-means++ initializations for stability...\n', ...
    Validation_nStabilityRuns);

for iRun = 1:Validation_nStabilityRuns
    rng(Validation_RandomSeed+iRun,'twister');

    [idx_run,~,sumd_run] = kmeans(RiskMapAD,nCluster, ...
        'Distance','sqeuclidean', ...
        'Start','plus', ...
        'Replicates',1, ...
        'MaxIter',1000, ...
        'Display','off');

    Validation_idxAll(:,iRun) = idx_run;
    Validation_SSE(iRun) = sum(sumd_run);
end

% Identify the repeated-initialization solution with the lowest SSE.
[Validation_BestSSE,Validation_BestRun] = min(Validation_SSE);
Validation_BestIdx = Validation_idxAll(:,Validation_BestRun);

% Quantify stability relative to the best-SSE repeated-initialization solution.
Validation_ARIvsBest = nan(Validation_nStabilityRuns,1);
for iRun = 1:Validation_nStabilityRuns
    Validation_ARIvsBest(iRun) = adjustedRandIndex( ...
        Validation_BestIdx,Validation_idxAll(:,iRun));
end

Validation_MeanARIvsBest = mean(Validation_ARIvsBest,'omitnan');
Validation_MedianARIvsBest = median(Validation_ARIvsBest,'omitnan');
Validation_MinARIvsBest = min(Validation_ARIvsBest);
Validation_MaxARIvsBest = max(Validation_ARIvsBest);

% Also compare the best-SSE validation solution with the ORIGINAL main-result
% clustering. This is descriptive only; the original clustering is retained.
Validation_ARI_BestVsOriginal = adjustedRandIndex(Original_idx,Validation_BestIdx);

fprintf('\nRepeated-initialization stability (k = %d):\n',nCluster);
fprintf('Best run (lowest SSE) = %d\n',Validation_BestRun);
fprintf('Lowest SSE            = %.6f\n',Validation_BestSSE);
fprintf('Mean ARI vs best-SSE solution   = %.4f\n',Validation_MeanARIvsBest);
fprintf('Median ARI vs best-SSE solution = %.4f\n',Validation_MedianARIvsBest);
fprintf('ARI range vs best-SSE solution  = %.4f to %.4f\n', ...
    Validation_MinARIvsBest,Validation_MaxARIvsBest);
fprintf('ARI: best-SSE solution vs ORIGINAL clustering = %.4f\n', ...
    Validation_ARI_BestVsOriginal);

figure('Color','w');
histogram(Validation_ARIvsBest,20);
xlabel('Adjusted Rand index vs best-SSE solution');
ylabel('Number of repeated initializations');
title('Stability of k = 3 clustering across repeated initialization');
saveas(gcf,[ValidationDir,filesep,'ARI_vs_BestSSE.png']);
savefig(gcf,[ValidationDir,filesep,'ARI_vs_BestSSE.fig']);

% Restore the original main-result variables explicitly before continuing
% to write cluster-center NIfTI files and classify MCI participants.
idx = Original_idx;
c = Original_c;


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


%% Local function: Adjusted Rand Index (ARI)
function ari = adjustedRandIndex(labels1,labels2)
labels1 = labels1(:);
labels2 = labels2(:);

[~,~,a] = unique(labels1);
[~,~,b] = unique(labels2);
cont = accumarray([a,b],1);

n = sum(cont(:));
comb2 = @(x) x.*(x-1)./2;

sumNij = sum(comb2(cont(:)));
sumAi = sum(comb2(sum(cont,2)));
sumBj = sum(comb2(sum(cont,1)));
totalPairs = n*(n-1)/2;

if totalPairs == 0
    ari = NaN;
    return;
end

expectedIndex = (sumAi*sumBj)/totalPairs;
maxIndex = 0.5*(sumAi+sumBj);
denominator = maxIndex-expectedIndex;

if abs(denominator) < eps
    ari = 1;
else
    ari = (sumNij-expectedIndex)/denominator;
end
end
