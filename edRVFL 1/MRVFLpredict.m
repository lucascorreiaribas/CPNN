function [TestingAccuracy,Testing_time,ProbScores] = MRVFLpredict(testX,testY,model,option)

[Nsample,Nfea] = size(testX);
activation = option.activation;

L = model.L;
w = model.w;
b= model.b;
beta = model.beta;
mu = model.mu;
sigma = model.sigma;


A = cell(L,1); %for L hidden layers
TestingAccuracy = zeros(L,1);
ProbScores = cell(L,1); %depends on number of hidden layer
A_input = testX;

tic
%% First Layer
for i = 1:L
    A1 = A_input * w{i}+ repmat(b{i},Nsample,1);
    if option.renormal == 1
        if option.normal_type ==0
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i});
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
            A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i});
        end
    end
    
%     A1 = A_input * w{i};
%     A1 = bsxfun(@rdivide,A1-repmat(mu{i},size(A1,1),1),sigma{i});
%     A1 = A1+ repmat(b{i},Nsample,1);
%     A1 = relu(A1);
    
%     A1 = A_input * w{i}+ repmat(b{i},Nsample,1);
%     A1 = sigmoid(A1);
    
    %A1_temp1 = [A_input,A1,ones(Nsample,1)]; 
    A1_temp1 = [testX,A1,ones(Nsample,1)]; 
    
    %A1_temp3 = A1; 
    A{i} = A1_temp1;
    %clear A1 A1_temp1 A1_temp2 w1 b1
    A_input = [testX A1];
    
    %% Calculate the testing accuracy
    beta_temp = beta{i};
    testY_temp = A1_temp1*beta_temp;
    
    %%MajorityVoting
%    [max_score,indx] = max(testY_temp,[],2);
%     pred_idx(:,i) = indx;

    %%softmax to generate probabilites
    testY_temp = bsxfun(@minus,testY_temp,max(testY_temp,[],2)); %for numerical stability
    prob_scores=softmax(testY_temp')';
    ProbScores{i} = prob_scores;
    
    %one layer's accuracy
%     [max_prob,indx] = max(prob_scores,[],2);
%     [~, ind_corrClass] = max(testY,[],2);
%     TestAccuracy = mean(indx == ind_corrClass);

    %%Calculate the testing accuracy for first i layers
    TestingAccuracy(i,1) = ComputeAcc(testY,ProbScores,i);
    %TestingAccuracy(i,1) = majorityVoting(testY,pred_idx);
    
end


Testing_time = toc;



end