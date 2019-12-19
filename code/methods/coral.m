function [acc] = coral(Xs,Ys,Xt,Yt)
%CORAL Summary of this function goes here
%   Detailed explanation goes here
    cov_source = cov(Xs) + eye(size(Xs, 2));
    cov_target = cov(Xt) + eye(size(Xt, 2));
    A_coral = cov_source^(-1/2)*cov_target^(1/2);
    Sim_coral = Xs * A_coral * Xt';
    Sim_Trn = Xs * A_coral *  Xs';
    index = [1:1:size(Sim_coral,1)]';
    x = [[1:1:size(Sim_coral,2)]' Sim_coral'];
    Sim_Trn = [index Sim_Trn ];
    model = svmtrain(full(Ys), Sim_Trn, sprintf('-t 4 -c 1 -q'));
    [label, acc,scores] = svmpredict(full(Yt),Sim_coral, model);
end

