clear all;

addpath(genpath('libsvm'));
addpath(genpath('methods'));
addpath(genpath('../non_deep_da'));

%% Office_Caltech_Surf Dataset
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.gamma = 1.0;         % width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.subspace_dim_d = 100;  %SA: Subspace Dimensions
options.landmarks = 125;     %NSO: Number of Landmarks

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
srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
testsize =20;
accuracy = [];
stdd = [];
time = [];
for iData = 1:12
    acc_data = [];
    time_data = [];
    for i=1:testsize
        src = char(srcStr{iData});
        tgt = char(tgtStr{iData});
        data = strcat(src, '_vs_', tgt);
        fprintf('data=%s\n', data);
        load(['../data/OfficeCaltech/surf/' src '_SURF_L10.mat']);
        Xs = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Ys = labels;
        Xs = zscore(Xs);

        
        load(['../data/OfficeCaltech/surf/' tgt '_SURF_L10.mat']);
        Xt = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Yt = labels;
        Xt = zscore(Xt);
      
        
        [acc_data,time_data] = approaches(Xs,Ys,Xt,Yt,options,acc_data,time_data);
    end
    acc_data = reshape(acc_data, [size(acc_data,2) / testsize, testsize])';
    time_data = reshape(time_data, [size(time_data,2) / testsize, testsize])';
    accuracy = [accuracy; mean(acc_data)];
    time = [time; mean(time_data)];
    stdd = [stdd; std(acc_data)];
    save('study_offficeCaltech_surf.mat','accuracy',"stdd","time")
end
save('study_offficeCaltech_surf.mat','accuracy',"stdd","time")