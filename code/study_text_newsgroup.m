clear all;

addpath(genpath('libsvm'));
addpath(genpath('methods'));
addpath(genpath('../non_deep_da'));
%% 20 Newsgroup
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.gamma = 1.0;         % width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.subspace_dim_d = 5;  %SA: Subspace Dimensions
options.landmarks = 500;    %NTVM: Number of Landmarks
% JGSA
options.k = 30;             % subspace base dimension
options.T = 10;             % #iterations, default=10
options.alpha= 1;           % the parameter for subspace divergence ||A-B||
options.mu = 1;             % the parameter for target variance
options.beta = 0.1;         % the parameter for P and Q (source discriminaiton)
options.gamma = 2;          % the parameter for kernel

% MEDA: Parameters
options.d = 20;
options.rho = 1.0;
options.p = 10;
options.lambda = 10.0;
options.eta = 0.1;
options.T = 10;
testsize = 20;
accuracy = [];
stdd = [];
time = [];

for name = {'comp_vs_rec','comp_vs_sci','comp_vs_talk','rec_vs_sci','rec_vs_talk','sci_vs_talk'}%
    acc_data = [];
    time_data = [];
    for j=[1:36]
        for i=1:testsize
            data = char(name);
            data = strcat(data, '_', num2str(j));
            load(strcat('../data/20Newsgroup/', data));
            fprintf('data=%s\n', data);
            Xs=zscore(Xs');
            Xt=zscore(Xt');
            
            soureIndx = crossvalind('Kfold', Ys, 2);
            targetIndx = crossvalind('Kfold', Yt,2);
            Xs = Xs(find(soureIndx==1),:);
            Ys = Ys(find(soureIndx==1),:);

            [acc_data,time_data] = approaches(Xs,Ys,Xt,Yt,options,acc_data,time_data);
     
        end
    end
    study_size = testsize*18;
    acc_data = reshape(acc_data, [size(acc_data,2) / study_size, study_size])';
    time_data = reshape(time_data, [size(time_data,2) / study_size, study_size])';
    accuracy = [accuracy; mean(acc_data)];
    time = [time; mean(time_data)];
    stdd = [stdd; std(acc_data)];
    save('study_text_newsgroup.mat','accuracy',"stdd","time")
end
save('study_text_newsgroup.mat','accuracy',"stdd","time")