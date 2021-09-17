    function [trainAcc,testAcc,ValtrainAcc_acc_temp_inner4fold,ValtestAcc_acc_temp_inner4fold,validation_trainAcc,validation_testAcc,option_best] = ... 
    ensemble(dataX,dataY,train_index,test_index,validation_train,validation_test,tune)

% tuning_range
L_1st = tune.L_1st;
L_2nd = tune.L_2nd;

N_range = tune.N_range;
N_max = sum(validation_train(:,1))+size(dataX,2);
if 1.2*max(N_range)>N_max
    N_range = [floor(N_max/4.8),floor(N_max/2.4),floor(N_max/1.2)];
end

C_range = tune.C_range;
activation_range = tune.activation_range;
if tune.pseudo == 1
    C_range = [C_range 10000000];
end

option.L = L_1st;

renormal_switch = 1;
if tune.renormal == 0
    renormal_switch = [renormal_switch 0];
end


%
num_CV = size(validation_train,2);
option_best = cell(1,1);
validation_trainAcc = zeros(1,1);
validation_testAcc = zeros(1,1);
ValtrainAcc_acc_temp_inner4fold = zeros(L_1st,4);
ValtestAcc_acc_temp_inner4fold = zeros(L_1st,4);

%% 1st tuning
MAX_acc = 0; 

for norm = 1:numel(renormal_switch)

    option.renormal = renormal_switch(norm);

    for n = 1:numel(N_range)

        option.N = N_range(n);

        for j = 1:numel(C_range)

            option.C = C_range(j);

            for act = 1:numel(activation_range)

                option.activation = activation_range(act);

                for type = 1:numel(tune.normal_type)

                    option.normal_type = tune.normal_type(type);


                    for i = 1:num_CV

                        trainX_val = dataX(validation_train(:,i),:);
                        trainY_val = dataY(validation_train(:,i),:);
                        testX_val = dataX(validation_test(:,i),:);
                        testY_val = dataY(validation_test(:,i),:);


                        [~,ValtrainAcc_acc_temp_inner4fold(:,i),ValtestAcc_acc_temp_inner4fold(:,i),~,~]  = ... 
                            MRVFL(trainX_val,trainY_val,testX_val,testY_val,option);
                        % [~,ValtrainAcc_acc_temp_inner4fold(:,i),ValtestAcc_acc_temp_inner4fold(:,i),~,~]  = ... 
                        %    MRVFL(zscore(trainX_val),trainY_val,zscore(testX_val),testY_val,option);

                    end

                    train_acc_temp = mean(ValtrainAcc_acc_temp_inner4fold,2);
                    test_acc_temp = mean(ValtestAcc_acc_temp_inner4fold,2);

                    for k = 2:L_1st
                        if test_acc_temp(k)>MAX_acc
                            validation_trainAcc(i) = train_acc_temp(k);
                            validation_testAcc(i) = test_acc_temp(k);
                            MAX_acc = test_acc_temp(k);
                            option_val.stage = '1st';
                            option_val.validation_trainAcc = validation_trainAcc(i);
                            option_val.validation_testAcc = validation_testAcc(i);
                            option_val.C = option.C;
                            option_val.N = option.N;
                            option_val.L = k;
                            option_val.activation = option.activation;
                            option_val.renormal = option.renormal;
                            option_val.normal_type = option.normal_type;
                            option_val.outterNum_CV = tune.num_CV;
                            option_best = option_val;
                            option_val
                        end
                    end
                end
            end
        end
    end
end

%% 2nd tuning

ValtrainAcc_acc_temp_inner4fold_second = zeros(L_2nd,4);
ValtestAcc_acc_temp_inner4fold_second = zeros(L_2nd,4);


if tune.second == 1 && option_best.C~=0
    
    option_temp = option_best;
    option_temp.L = L_2nd;
    C_temp = log2(option_best.C);
    C_range_new = (C_temp - 2):1:(C_temp + 2);
    N_range_new = round([0.8,0.9,1,1.1,1.2].*option_best.N);
    
    MAX_acc = 0;
    
    for n = 1:numel(N_range_new)

        option_temp.N = N_range_new(n);
    
        for j = 1:numel(C_range_new)

            option_temp.C = 2^(C_range_new(j));

            for i = 1:num_CV

                trainX_val = dataX(validation_train(:,i),:);
                trainY_val = dataY(validation_train(:,i),:);
                testX_val = dataX(validation_test(:,i),:);
                testY_val = dataY(validation_test(:,i),:);


                [~,ValtrainAcc_acc_temp_inner4fold_second(:,i),ValtestAcc_acc_temp_inner4fold_second(:,i),~,~]  = ... 
                    MRVFL(trainX_val,trainY_val,testX_val,testY_val,option_temp);

            end

            train_acc_temp = mean(ValtrainAcc_acc_temp_inner4fold_second,2);
            test_acc_temp = mean(ValtestAcc_acc_temp_inner4fold_second,2);


            for k = 2:L_2nd
                if test_acc_temp(k)>MAX_acc
                    validation_trainAcc(i) = train_acc_temp(k);
                    validation_testAcc(i) = test_acc_temp(k);
                    MAX_acc = test_acc_temp(k);
                    option_val.stage = '2nd';
                    option_val.validation_trainAcc = validation_trainAcc(i);
                    option_val.validation_testAcc = validation_testAcc(i);
                    option_val.C = option_temp.C;
                    option_val.N = option_temp.N;
                    option_val.L = k;
                    option_val.activation = option_temp.activation;
                    option_val.renormal = option_temp.renormal;
                    option_val.normal_type = option.normal_type;
                    option_val.outterNum_CV = tune.num_CV;
                    option_best = option_val;
                    option_val
                end
            end
        end
    end
end
    

%% train and test

trainX = dataX(train_index,:);
trainY = dataY(train_index,:);
testX = dataX(test_index,:);
testY = dataY(test_index,:);
    
[~,trainAcc_temp,testAcc_temp,~,~]  = MRVFL(trainX,trainY,testX,testY,option_best);
trainAcc = trainAcc_temp(option_best.L);
testAcc = testAcc_temp(option_best.L);


end
