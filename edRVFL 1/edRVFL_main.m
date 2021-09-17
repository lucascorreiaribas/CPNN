clear
clc

method = 'edRVFL';


dataset_path = 'UCI/'; % Need to change



[name,num] = GetFiles(dataset_path);
n_folders = num(1);


for dataset_num = 1:n_folders
    dataset_name = name(dataset_num).name;
    dataset_name
    dataset_path_name = strcat(dataset_path,dataset_name,'/');
    load([dataset_path_name 'folds.mat']);
    load(strcat(dataset_path_name,dataset_name, '.mat'))
    load([dataset_path_name 'labels.mat']);
    load([dataset_path_name 'validation_train.mat']);
    load([dataset_path_name 'validation_test.mat']);
    load([dataset_path_name 'numOffold.mat']);
    numOffold = numOffold; %4-fold CV
    expression = strcat('dataX = ',dataset_name,';');

    eval(expression);
    dataY=labels;
    folds = logical(folds);
    validation_train = logical(validation_train);
    validation_test = logical(validation_test);

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
    tune.C_range = C_range;
    tune.C_rate_range = C_rate_range;
    tune.activation_range = activation_range;
    tune.L_1st = L_1st;
    tune.L_2nd = L_2nd;
    tune.second = 1;
    tune.pseudo = 0;
    tune.renormal = 1;
    tune.normal_type = 1;

   

    %results
    validation_trainAcc = cell(numOffold,1);
    validation_testAcc = cell(numOffold,1);
    trainAcc = zeros(numOffold,1);
    testAcc = zeros(numOffold,1);
    option_save = cell(numOffold,1);
    ValtrainAcc_single = cell(numOffold,1);
    ValtestAcc_single = cell(numOffold,1);

    %%outer crossvalidation
    for num_CV=1:numOffold

        tune.num_CV = num_CV;

        test_index = folds(:,num_CV);
        train_index = logical(1-test_index);
        validation_train_index = validation_train(:,(num_CV-1)*4+1:num_CV*4);
        validation_test_index = validation_test(:,(num_CV-1)*4+1:num_CV*4);

        [trainAcc(num_CV),testAcc(num_CV),ValtrainAcc_single{num_CV},ValtestAcc_single{num_CV},validation_trainAcc{num_CV},validation_testAcc{num_CV},option_save{num_CV}] = ... 
        ensemble(dataX,dataY_temp,train_index,test_index,validation_train_index,validation_test_index,tune);
    end

    mean(trainAcc)
    mean(testAcc)

    %% save

    results.validation_trainAcc = validation_trainAcc;
    results.validation_testAcc = validation_testAcc;
    results.trainAcc = trainAcc;
    results.testAcc = testAcc;
    results.option_save = option_save;
    results.ValtrainAcc_single = ValtrainAcc_single;
    results.ValtestAcc_single = ValtestAcc_single;

end

function [names,class_num] = GetFiles(dataset_path)
files = dir(dataset_path);
size0 = size(files);
length = size0(1);
names = files(1:length);
class_num = size(names);
end