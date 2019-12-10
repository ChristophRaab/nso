function [acc] = gca123(Xs,Ys,Xt,Yt,t,w1,w2)
%GCA Summary of this function goes here
%   Detailed explanation goes here
    n = size(Xs, 1); 
    m = size(Xt,1); 
    cOpts.UseFastNNSearch = 1; 
    % compute graph Laplacian kernels for source and target domains
    [vGraph_s, vEigenVecsSub_s, vEigenVals_s, vOpts, vLaplacian_s, vNNInfo_s,vDiffusion_s,vQ_s] = FastLaplacianDetEigs( Xs, cOpts );
    [vGraph_t, vEigenVecsSub_t, vEigenVals_t, vOpts, vLaplacian_t, vNNInfo_t,vDiffusion_t,vQ_t] = FastLaplacianDetEigs( Xt, cOpts );
    
    % compute source and target graph kernels using manifold diffusion for
    % now
    kernel_s = zeros(size(vEigenVecsSub_s,2)); 
    
    for i=1:size(vEigenVecsSub_s,1) 
        kernel_s = kernel_s + exp(-1/2*vEigenVals_s(i))*vEigenVecsSub_s(i,:)'*vEigenVecsSub_s(i,:); 
    end
    
    kernel_t = zeros(size(vEigenVecsSub_t,2)); 
    
    for i=1:size(vEigenVecsSub_t,1) 
        kernel_t = kernel_t + exp(-1/2*vEigenVals_t(i))*vEigenVecsSub_t(i,:)'*vEigenVecsSub_t(i,:); 
    end
    
    invK = zeros(n+m); 
    
    invK(1:n, 1:n) = inv(kernel_s + eye(n)); % to ensure SPD 
    invK(n+1:m+n, n+1:n+m) = inv(kernel_t + eye(m)); 
    
    
    cov_source = cov(Xs) + eye(size(Xs,2)); 
    cov_target = cov(Xt) + eye(size(Xt,2)); 
    
    % form MMD L matrix to minimize distributional differences between
    % source and target
    
    Xall = [Xs; Xt]; 
    
    Lmat = ones(m + n)*(-1/n*m); 
    
    Lmat(1:n,1:n) = ones(n,n)*(1/n^2); 
    Lmat(n+1:m+n,n+1:m+n) = ones(m,m)*(1/m^2); 
    
    
    A_gca_inv_source = sharp(pinv(cov_source + Xall'*(w1*Lmat + w2*invK)*Xall),cov_target, t); 
    
    Sim = Xs * A_gca_inv_source * Xt';
%     accy_coral_mda(iter) = SVM_Accuracy(Xs, A_gca_inv_source, Yt, Sim_gca_inv_source, Ys);
    Sim_Trn = Xs * A_gca_inv_source *  Xs';
    index = [1:1:size(Sim,1)]';
    Sim = [[1:1:size(Sim,2)]' Sim'];
    Sim_Trn = [index Sim_Trn ];
    model = svmtrain(Ys, Sim_Trn, sprintf('-t 4 -c 1 -q'));
    [label, acc, scores] = svmpredict(Yt, Sim, model);
end

