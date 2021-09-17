function [ff] = CPNN(video,QS,R)

%% CPNN dynamic texture descriptor
% this code computes a feature vector using the CPNN descriptor
% input:
%   -video: a 3D matrix with HxWxN pixels
%   -QS: set of number of hidden neurons (default: QS = [4,24,29])
%   -R: set of maximum radius for modeling (default: R = [4,10];
% output:
%   -ff: feature vector
%
%   cite: Ribas, C.L., Sá Junior, J.J.M., Manzanera, A. and Bruno, O.M..
%   Learning Graph Representation with Randomized Neural Network for Dynamic Texture Classification.
%%

KS = []; KT = []; FS = []; FT = []; FS_IN = []; FT_IN = []; D =[];
for r = 1:max(R)
    [KS(r,:) KT(r,:) FS(r,:) FT(r,:) FS_IN(r,:) FT_IN(r,:) D] = CNModelVideo(double(video),r,size(video,3));
end


featuresKS = []; featuresKT = []; featuresFS = []; featuresFT = [];
x = 1;
for r = 1:length(R)
    for Q = 1:length(QS)
        featuresKS = [featuresKS ELM(KS(1:R(r),:),D,QS(Q))];
        featuresKT = [featuresKT ELM(KT(1:R(r),:),D,QS(Q))];
        featuresFS = [featuresFS ELM(FS(1:R(r),:),D,QS(Q))];
        featuresFT = [featuresFT ELM(FT(1:R(r),:),D,QS(Q))];
    end
end

ff = [featuresKS, featuresKT, featuresFS, featuresFT];

end

function [M] = ELM(X,D,Q)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ELM neural network by Lucas Ribas
% Input:
%        X: features input [P,N], N number of samples and P number of
% inputs or features
%        D: vector of class [1xN]
%        Q: number of neurons of the hidden layear
%
% Output:
%        M: matrix of weights [1 x (Q+1)] as signature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[P N] = size(X); %P is the number of input and N the number of samples

%normalization Z-Score
X = zscore(X');
X = X';
W = LCG(Q,P+1,Q*(P+1)); % matrix of weights
X = [X;-ones(1,N)]; %add bias

Z = g(W*X); %function of activation of hidden layear
Z = [Z -ones(1,N)']'; %add bias hidden output

lambda = 0.001;
M = (D*Z')/(Z*Z'+ lambda * eye(Q+1)); % weights of output layear
end

%% sigmoide fuction
function y = g(u)
y = 1./(1+exp(-u))';
end

%% LCG Random
function Mx = LCG(m,n,L)
V(1)=L+1;
a = L+2;
b = L+3;
c = L^2;
for x=1:(m*n)-1
    V(x+1) = mod((a*V(x)+b),c);
end

V = zscore(V);
Mx = reshape([V(:) ; zeros(rem(n - rem(numel(V),n),n),1)],n,[]).';
end