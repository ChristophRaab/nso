clear all;

addpath(genpath('libsvm'));
addpath(genpath('methods'));
addpath(genpath('../non_deep_da'));



%% Reuters Dataset
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.gamma = 1.0;         % width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.subspace_dim_d = 10;  %SA: Subspace Dimensions
options.ntvm_ker = 'rbf';     %NTVM: Kernel Type
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

for strData = {'org_vs_people','org_vs_place', 'people_vs_place'} %
    
    for iData = 1:2
    acc_data = [];
    time_data = [];
        for i=1:testsize
            
            data = char(strData);
            data = strcat(data, '_', num2str(iData));
            load(strcat('../data/Reuters/', data));
            
            fprintf('data=%s\n', data);
            Xs=zscore(Xs');
            Xt=zscore(Xt');
            Xs = normr(Xs);
            Xt = normr(Xt);
            [acc_data,time_data] = approaches(Xs,Ys,Xt,Yt,options,acc_data,time_data);
        end
        acc_data = reshape(acc_data, [size(acc_data,2) / testsize, testsize])';
        time_data = reshape(time_data, [size(time_data,2) / testsize, testsize])';
        accuracy = [accuracy; mean(acc_data)];
        time = [time; mean(time_data)];
        stdd = [stdd; std(acc_data)];
        save('study_text_reuters.mat','accuracy',"stdd","time")
    end
end
save('study_text_reuters.mat','accuracy',"stdd","time")