
%for mac and linux

video = randi(255,[50,50,50]); % load a video 


Q = [4,24,29]; % number of hidden neurons
radius = [4,10]; % maximum radius

features = CPNN(video,Q,radius); %compute the feature vector

