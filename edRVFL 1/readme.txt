link for edRVFL: https://github.com/P-N-Suganthan/CODES

Some parts of the codes have been taken from:
1. Zhang, Le, and Ponnuthurai N. Suganthan. "A comprehensive evaluation of random vector functional link networks." Information sciences 367 (2016): 1094-1105.
2. Tang, Jiexiong, Chenwei Deng, and Guang-Bin Huang. "Extreme learning machine for multilayer perceptron." IEEE transactions on neural networks and learning systems 27.4 (2015): 809-821.
3. Katuwal, Rakesh, and Ponnuthurai N. Suganthan. "Stacked autoencoder based deep random vector functional link neural network for classification." Applied Soft Computing 85 (2019): 105854.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
We have put a demo of edRVFL with arrhythmia dataset 

The following is the parameters setting for two-stage tuning

L_1st = 3;  %layer number in 1st tuning 
L_2nd = 10;  %layer number in 2nd tuning 
N_range = [256,512,1024]; % Hidden neuron number range
C_range =2.^(-6:3:12);  % Regalurazation parameter range
C_rate_range = [0.5,0.8,1,1.2,1.5];  % Fine-tune rate in the 2nd tuning
activation_range = 1:3; %(1)selu;(2)relu;(3)sigmoid;(4)sign;(5)hardlim;(6)tribas;(7)radbas;(8)sign.

tune.N_range = N_range;
tune.C_range = C_range;
tune.C_rate_range = C_rate_range;
tune.activation_range = activation_range;
tune.L_1st = L_1st;
tune.L_2nd = L_2nd;
tune.second = 1;  %Whether to use two-stage tuning
tune.pseudo = 0;  %Whether to use psedo inverse
tune.renormal = 1;  %Whether to do renormalization
tune.normal_type = 1;  %Renormalization type (0 or 1)

The main steps of the two-stage tuning can be listed as follows: 
Step 1) Fix the number of layers to 3, and then select the optimal number of neurons (N*) and regularization parameter (C*) based on a coarse range for N and C. 
Step 2) Tune the number of layers from 2 to the maximum and fine-tune the N, C parameters by considering only a fine range in the neighborhood of N* and C*. 
It is also worth mentioning that in the first stage, we also choose the activation function among (relu, selu, sigmoid) with the best performance on the validation set.

During the tuning process, we do 4-fold cross-validation to find the best parameter settings. 
25% of the training data are used as the validation set, and we select the hyperparameters with the best average validation accuracy. 
Then, we use the whole training data to train the models before feeding the test data into them. 
At last, the testing accuracy is obtained based on the correct predictions of the networks for the test data.


The codes are not optimized for efficiency. The codes have been cleaned for better readability
and documented and are not exactly the same as used in our paper. We have re-run and checked the
codes only in few datasets so if you find any bugs/issues, please write to
Qiushi (qiushi001@e.ntu.edu.sg).


03-Apr-2021