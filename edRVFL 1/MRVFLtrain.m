function [model,TrainingAccuracy,Training_time,ProbScores] = MRVFLtrain(trainX,trainY,option)

[Nsample,Nfea] = size(trainX);
N = option.N;
L = option.L;
C = option.C;
activation = option.activation;
s = 1;  %scaling factor


A = cell(L,1); %for L hidden layers
beta = cell(L,1);
weights = cell(L,1);
biases = cell(L,1);
mu = cell(L,1);
sigma = cell(L,1);
TrainingAccuracy = zeros(L,1);
ProbScores = cell(L,1); %depends on number of hidden layer

A_input = trainX;

tic
for i = 1:L
    
    if i==1
        w = s*2*rand(Nfea,N)-1;
    else
        w = s*2*rand(Nfea+N,N)-1;
    end
    
    b = s*rand(1,N);
    weights{i} = w;
    biases{i} = b;
    
    A1 = A_input * w+repmat(b,Nsample,1);
    if option.renormal == 1
        if option.normal_type ==0
            mu{i} = mean(A1,1);
            sigma{i} = std(A1);
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
        end
    end  
    if activation == 1
        A1 = selu(A1);
    elseif activation == 2
        A1 = relu(A1);
    elseif activation == 3
        A1 = sigmoid(A1);
    elseif activation == 4
        A1 = sin(A1);
    elseif activation == 5
        A1 = hardlim(A1);        
    elseif activation == 6
        A1 = tribas(A1);
    elseif activation == 7
        A1 = radbas(A1);
    elseif activation == 8
        A1 = sign(A1);
    elseif activation == 9
        A1 = swish(A1);
    end
    if option.renormal == 1
        if option.normal_type ==1
            mu{i} = mean(A1,1);
            sigma{i} = std(A1);
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;layer normalization
        end
    end
    
%     A1 = A_input * w;
%     mu{i} = mean(A1,1);
%     sigma{i} = std(A1);
%     A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i}); %;batch normalization
%     A1 = A1+repmat(b,Nsample,1);
%     A1 = relu(A1);
    
%     A1 = A_input * w+repmat(b,Nsample,1);
%     A1 = sigmoid(A1);
    
    %A1_temp1 = [A_input,A1,ones(Nsample,1)];
    A1_temp1 = [trainX,A1,ones(Nsample,1)];
    beta1  = l2_weights(A1_temp1,trainY,C,Nsample);
    
    A{i} =  A1_temp1;
    beta{i} = beta1;
    
    %clear A1 A1_temp1 A1_temp2 beta1
    A_input = [trainX A1];
    
    
   %% Calculate the training accuracy
    trainY_temp = A1_temp1*beta1;
    
    %MajorityVoting
%     [max_score,indx] = max(trainY_temp,[],2);
%     pred_idx(:,i) = indx;

    %Softmax to generate probabilites
    trainY_temp = bsxfun(@minus,trainY_temp,max(trainY_temp,[],2)); %for numerical stability
    prob_scores=softmax(trainY_temp')';
    ProbScores{i,1} = prob_scores;

    %One layer's accuracy
%     [~,indx] = max(prob_scores,[],2);
%     [~, ind_corrClass] = max(trainY,[],2);
%     correct_index{i} = (indx == ind_corrClass);

    
    %Calculate the training accuracy for first i layers
    TrainingAccuracy(i,1) = ComputeAcc(trainY,ProbScores,i); %averaging prob.scores
    %TrainingAccuracy(i,1) = majorityVoting(trainY,pred_idx); %majority voting
end

Training_time = toc;


%%
model.L = L;
model.w = weights;
model.b = biases;
model.beta = beta;
model.mu = mu;
model.sigma = sigma;

end