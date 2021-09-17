function acc = majority(Y,ProbScores,option_best)



for i = 1:4  
    Class_dist = zeros(size(Y));
    for j=1:option_best{i}.L
      Dist_temp = ProbScores{i}{j,1};  
      Class_dist = Class_dist + Dist_temp;
    end
    AvgProbScores = Class_dist./length(ProbScores);
    [~,indx] = max(AvgProbScores,[],2);
    pred_idx(:,i) = indx;   
end
    
acc = majorityVoting(Y,pred_idx); %majority voting

end