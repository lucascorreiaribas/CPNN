clear
clc

method = 'edRVFL';

dataset = 'ucla9'; %use ucla9 or ucla8

load(['cpnn/CPNN_',dataset,'.mat']);

if strcmp(dataset, 'ucla8') == 1
    type = 'holdout';
    k = 0.5;
    numRep = 20;
    features = data8(:,1:end-1);
    classes = data8(:,end);
elseif strcmp(dataset, 'ucla9') == 1
    type = 'holdout';
    k = 0.5;
    numRep = 20;
    features = data9(:,1:end-1);
    classes = data9(:,end);
end



n_folders = k;
numOffold = k;



dataY=classes;


U_dataY = unique(dataY);
nclass = numel(U_dataY);
dataY_temp = zeros(numel(dataY),nclass);

% 0-1 coding for the target
for i=1:nclass
    idx = dataY==U_dataY(i);
    dataY_temp(idx,i)=1;
end

L_1st = 3;
L_2nd = 10;
N_range = [256,512,1024];
C_range =2.^(-6:3:12);
C_rate_range = [0.5,0.8,1,1.2,1.5];
activation_range = 1:3; %(1)selu;(2)relu;(3)sigmoid;(4)sign;(5)hardlim;(6)tribas;(7)radbas;(8)sign.

tune.N_range = N_range;
tune.C_range = [0.001,0.005,0.0001,0.015,0.01,0.2,0.1,0.5];
tune.C_rate_range = C_rate_range;
tune.activation_range = activation_range;
tune.L_1st = L_1st;
tune.L_2nd = L_2nd;
tune.second = 1;
tune.pseudo = 0;
tune.renormal = 1;
tune.normal_type = 1;




num_CV=1;
accuracies = [];
for rep = 1:numRep
    rng(rep);
    cvp = cvpartition(classes,'Holdout',k);
    
    trainAcc = []; testAcc = [];
    
    tune.num_CV = 1;
    
    trainY = classes(cvp.training);
    testY = classes(cvp.test);
    train_index = cvp.training;
    test_index = cvp.test;
    
    train_numIndex = find(cvp.training==1);
    
    validation_test_index = zeros(length(classes),1);
    validation_train_index = zeros(length(classes),1);
    cvp2 = cvpartition(trainY,'Holdout',0.1);
    
    validation_test_index(train_numIndex(cvp2.test)) = 1;
    validation_train_index(train_numIndex(cvp2.training)) = 1;
    
    
    validation_train_index = logical(validation_train_index);
    validation_test_index = logical(validation_test_index);
    
    
    [trainAcc(num_CV),testAcc(num_CV),ValtrainAcc_single{num_CV},ValtestAcc_single{num_CV},validation_trainAcc{num_CV},validation_testAcc{num_CV},option_save{num_CV}] = ...
        ensemble(features,dataY_temp,train_index,test_index,validation_train_index,validation_test_index,tune);
    
    
    mean(trainAcc);
    mean(testAcc);
    accuracies(rep) = mean(testAcc);
end


disp(['Average Accuracy (%) for ',dataset,': ',num2str(mean(accuracies))]);

avg_acc = mean(accuracies);
save(['acc_',dataset,'.mat'],'accuracies','avg_acc');

%Average Accuracy (%) for ucla9: 0.93214
%Average Accuracy (%) for ucla8: 0.17045
