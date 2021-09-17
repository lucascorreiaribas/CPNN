function [Model,TrainAcc,TestAcc,TrainingTime,TestingTime] = ...
    MRVFL(trainX,trainY,testX,testY,option)

seed = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(seed);

% Train RVFL
[Model,TrainAcc,TrainingTime,~] = MRVFLtrain(trainX,trainY,option);

% Using trained model, predict the testing data
[TestAcc,TestingTime,~] = MRVFLpredict(testX,testY,Model,option);

end
%EOF