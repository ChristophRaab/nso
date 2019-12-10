%% Matlab Demo file For Nystr�m Basis Transfer
% Sampling as suggested in
% B. Gong, Y. Shi, F. Sha and K. Grauman, "Geodesic flow kernel for unsupervised domain adaptation," 2012 IEEE Conference on Computer Vision and Pattern Recognition, Providence, RI, 2012,
% doi: 10.1109/CVPR.2012.6247911
%%
clear all;

addpath(genpath('/libsvm'));
addpath(genpath('../data'));
addpath(genpath('../code'));

testruns = 5;
%% OFFICE vs CALLTECH-256 Dataset
options.ker = 'rbf';         % TKL: kernel: 'linear' | 'rbf' | 'lap'
options.eta = 1.1;           % TKL: eigenspectrum damping factor
options.gamma = 1.0;         % TKL: width of gaussian kernel
options.g = 65;              % GFK: subspace dimension
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.tcaNv = 60;          % TCA: numbers of Vectors after reduction
options.theta = 1;          %PCVM: Width of gaussian kernel
options.landmarks = 250;     %NTVM: Number of Landmark
options.subspace_dim_d = 5;  %SA: Subspace Dimensions
srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
result = [];
for iData = 1:12
    acc_data = [];
    for i = 1:testruns
        acc_run=[];
        src = char(srcStr{iData});
        tgt = char(tgtStr{iData});
        data = strcat(src, '_vs_', tgt);
        fprintf('data=%s\n', data);
        load(['../data/OfficeCaltech/' src '_SURF_L10.mat']);
        fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
        Xs = zscore(fts, 1);
        Ys = labels;
        Xs = full(Xs);
        
        load(['../data/OfficeCaltech/' tgt '_SURF_L10.mat']);
        fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
        Xt = zscore(fts, 1);
        Yt = labels;
        Xt = full(Xt);
        
        m = size(Xs, 2);
        n = size(Xt, 2);
        
        inds = split(Ys, 20);
        Xs = Xs(inds,:)';
        Xt = Xt';
        Ys = Ys(inds);
        m = size(Xs, 2);
        n = size(Xt, 2);
        %% SVM
        K = kernel(options.ker, [Xs, Xt], [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('SVM = %.2f%%\n', acc(1));
        acc_run = [acc_run acc(1)];
        
        %% TCA
        nt = length(Ys);
        mt = length(Yt);
        K = tca(Xs',Xt',options.tcaNv,options.gamma,options.ker);
        model = svmtrain(full(Ys),[(1:nt)',K(1:nt,1:nt)],['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label,acc,scores] = svmpredict(full(Yt),[(1:mt)',K(nt+1:end,1:nt)],model);
        
        fprintf('\nTCA %.2f%% \n',acc(1));
        acc_run = [acc_run acc(1)];
        %% JDA
        Cls = [];
        [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        K = kernel(options.ker, Z, [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\nJDA %.2f%% \n',acc(1));
        acc_run = [acc_run acc(1)];
        
        %% TKL SVM
        tic;
        K = TKL(Xs, Xt, options);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
        [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\nTKL %.2f%%\n',acc(1));
        acc_run = [acc_run acc(1)];
        %% GFK
        xs = full(Xs');
        xt = full(Xt');
        Ps = pca(xs);
        Pt = pca(xt);
        nullP = null(Ps');
        G = GFK([Ps,nullP], Pt(:,1:options.g));
        [label, acc] = my_kernel_knn(G, xs, Ys, xt, Yt);
        acc = full(acc)*100;
        fprintf('\n GFK %.2f%%\n',acc);
        acc_run = [acc_run acc];
        %% SA
        [Xss,~,~] = pca(Xs');
        [Xtt,~,~] = pca(Xt');
        Xss = Xss(:,1:options.subspace_dim_d);
        Xss = Xtt(:,1:options.subspace_dim_d);
        [predicted_label, acc, decision_values,model] =  Subspace_Alignment(Xs',Xt',Ys,Yt,Xss,Xtt);
        fprintf('\nSA %.2f%%\n',acc(1));
        acc_run = [acc_run acc(1)];
%         %% CGCA
%         acc= gca123(Xs',Ys,Xt',Yt,0.9, 0.2, 0.1)
%         fprintf('\nCGCA %.2f%% \n',acc);
%         acc_run = [acc_run acc(1)];
        %% Coral
        acc = coral(Xs',Ys,Xt',Yt);
        fprintf('\n: Coral %.2f%% \n', acc(1));
        acc_run = [acc_run acc(1)];
        %% NSO
        [model,K,m,n] = NSO(Xs',Xt',Ys,options);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        acc_run = [acc_run acc(1)];
        fprintf('\n: NSO %.2f%% \n', acc(1));
        acc_data = [acc_data; acc_run];
    end
    result = [result; mean(acc_data)];
end


function [idx1 idx2] = split(Y,nPerClass, ratio)
% [idx1 idx2] = split(X,Y,nPerClass)
idx1 = [];  idx2 = [];
for C = 1 : max(Y)
    idx = find(Y == C);
    rn = randperm(length(idx));
    if exist('ratio')
        nPerClass = floor(length(idx)*ratio);
    end
    idx1 = [idx1; idx( rn(1:min(nPerClass,length(idx))) ) ];
    idx2 = [idx2; idx( rn(min(nPerClass,length(idx))+1:end) ) ];
end
end