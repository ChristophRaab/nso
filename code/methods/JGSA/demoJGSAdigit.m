% Joint Geometrical and Statistical Alignment for Visual Domain Adaptation.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
% Jing Zhang, Wanqing Li, Philip Ogunbona.

clear;close all;
datapath = './data/';
% Set algorithm parameters
options.k = 100;            % #subspace bases 
options.ker = 'primal';     % kernel type, default='linear' options: linear, primal, gauss, poly

options.T = 10;             % #iterations, default=10

options.alpha= 1;           % the parameter for subspace divergence ||A-B||
options.mu = 1;             % the parameter for target variance
options.beta = 0.01;        % the parameter for P and Q (source discriminaiton) 
options.gamma = 2; 

dataStr = {'MNIST_vs_USPS','USPS_vs_MNIST'};


results = [];
for iData = 1:2
    load([datapath  char(dataStr{iData}) '.mat']);
    Xs = X_src;
    Xt = X_tar;

    Ys = Y_src;
    Yt = Y_tar;
    
    Xs = normc(Xs);
    Xt = normc(Xt);
    
    Cls = knnclassify(Xt',Xs',Ys,1); 
    acc = length(find(Cls==Yt))/length(Yt); 
    fprintf('acc=%0.4f\n',full(acc));

    Yt0 = Cls;
    [Zs, Zt, A, Att] = JGSA(Xs, Xt, Ys, Yt0, Yt, options);
    Cls = knnclassify(Zt',Zs',Ys,1); 
    acc = length(find(Cls==Yt))/length(Yt); 
    results = [results;acc];
end

