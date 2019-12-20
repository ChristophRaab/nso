function [model,K,m,n] = nso(Xs,Xt,Ys,options)
%Nystroem Subspace Override. Approximation of Subpsace via Nystroem
if exist('options', 'var')
    if ~isfield(options,"ker")
        options.ker = 'rbf';
    end
    if ~isfield(options, 'gamma')
        options.gamma = 1.0;
    end
    if ~isfield(options,'landmarks')
        options.landmarks = 500;
    end
    if ~isfield(options, "svmc")
        options.svmc = 10;
    end
else
    options.ker = 'rbf';
    options.gamma = 1.0;
    options.landmarks = 500;
    options.svmc = 10;
end

[Xs,Ys] = augmentation(Xs,size(Xt,1),Ys);
[Xt,Xs]=pd_ny_svd(Xt,Xs,Ys,options.landmarks);
Xs = zscore(Xs,1);
Xt = zscore(Xt,1);
m = size(Xs, 1);
n = size(Xt, 1);
K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
end

