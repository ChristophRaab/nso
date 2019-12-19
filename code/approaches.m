function [acc_data,time_data] = approaches(Xs,Ys,Xt,Yt,options,acc_data,time_data)
%APPROACHES: List of all domain adaptation methods list used in the study

    Xt = full(Xt); Xs = full(Xs); Ys = full(Ys); Yt = full(Yt);
    %
    Xt = zscore(Xt')'; Xs = zscore(Xs')';
    m = size(Xs, 1);
    n = size(Xt, 1);
    %% SVM
    tic
    K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
    model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
    [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    time_data = [time_data toc];
    fprintf('SVM = %.2f%%\n', acc(1));
    acc_data = [acc_data acc(1)];



    %% TCA
    tic
    nt = length(Ys);
    mt = length(Yt);
    K = tca(Xs,Xt,options.tcaNv,options.gamma,options.ker);
    model = svmtrain(full(Ys),[(1:nt)',K(1:nt,1:nt)],['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
    [label,acc,scores] = svmpredict(full(Yt),[(1:mt)',K(nt+1:end,1:nt)],model);
    time_data = [time_data toc];
    acc_data = [acc_data acc(1)];
    fprintf('\nTCA %.2f%% \n',acc(1));

    %% JDA
    tic;
    Cls = [];
    [Z,A] = JDA(Xs',Xt',Ys,Cls,options);
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
    Zs = Z(:,1:size(Xs',2));
    Zt = Z(:,size(Xs',2)+1:end);
    K = kernel(options.ker, Z, [],options.gamma);
    model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
    [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    time_data = [time_data toc];
    fprintf('\nJDA %.2f%% \n',acc(1));
    acc_data = [acc_data acc(1)];

    %% TKL SVM
    tic;
    K = TKL(Xs, Xt, options);
    model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
    [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    time_data = [time_data toc];
    fprintf('\nTKL %.2f%%\n',acc(1));
    acc_data = [acc_data acc(1)];

    %% GFK
    tic;
    Ps = pca(Xs);  % source subspace
    Pt = pca(Xt);  % target subspace
    G = GFK([Ps,null(Ps)], Pt(:,1:options.g));
    [~, acc] =my_kernel_knn(G, Xs, Ys, Xt, Yt);
    time_data = [time_data toc];
    fprintf('\nGFK %.2f%%\n',acc*100);
    acc_data = [acc_data acc*100];

    %% SA
    tic;
    [Xss,~,~] = pca(Xs);
    [Xtt,~,~] = pca(Xt);
    Xss = Xss(:,1:options.subspace_dim_d);
    Xss = Xtt(:,1:options.subspace_dim_d);
    [predicted_label, acc, decision_values,model] =  Subspace_Alignment(Xs,Xt,Ys,Yt,Xss,Xtt);
    time_data = [time_data toc];
    fprintf('\nSA %.2f%%\n',acc(1));
    acc_data = [acc_data acc(1)];


    %% Coral
    tic;
    acc = coral(Xs,Ys,Xt,Yt);
    time_data = [time_data toc];
    fprintf('\n Coral %.2f%% \n', acc(1));
    acc_data = [acc_data acc(1)];


    %% CGCA
    Xs_gca=zscore(full(Xs)); % Make sure that matrices are column and row normalized
    Xt_gca=zscore(full(Xt));
    tic;
    acc= gca123(Xs_gca,Ys,Xt_gca,Yt,0.9, 0.2, 0.1)
    time_data = [time_data toc];
    acc_data = [acc_data acc(1)];

    %% SCA
    tic;
    params.verbose = false;
    [acc, predicted_labels, Zs, Zt] = SCA(Xs,Ys, Xt, Yt, params);
    time_data = [time_data toc];
    acc_data = [acc_data acc];


    %% EasyTL
    tic;
    [acc, y] = EasyTL(Xs,Ys,Xt,Yt,'raw');
    fprintf('\n EasyTL %.2f%% \n', acc);
    time_data = [time_data toc];
    acc_data = [acc_data acc];

    %% JGSA
    tic;
    mdl = fitcknn(Xs,Ys);
    [Cls,score,cost] = predict(mdl,Xt);
    acc = length(find(Cls==Yt))/length(Yt);

    Yt0 = Cls;
    [Zs, Zt, A, Att] = JGSA(normr(Xs)',normr(Xt)', Ys, Yt0, Yt, options);
    mdl = fitcknn(Zs',Ys);
    [Cls,score,cost] = predict(mdl,Zt');
    acc = length(find(Cls==Yt))/length(Yt);
    fprintf('\n JGSA %.2f%% \n', acc);
    time_data = [time_data toc];
    acc_data = [acc_data acc];



    %% MEDA
    tic;
    [acc,~,~,Yt_pred] = MEDA(Xs,Ys,Xt,Yt,options);
    fprintf('\n Meda %.2f%% \n', acc);
    time_data = [time_data toc];
    acc_data = [acc_data acc];

    % SO
    tic;
    [model,K,m,n] = so(Xs',Xt',Ys,options);
    [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    fprintf('\n PD_NBT %.2f%% \n', acc(1));
    time_data = [time_data toc];
    acc_data = [acc_data acc(1)];

    %%     NBT classwise
    tic;
    [model,K,m,n] = pd_cl_nbt(Xs,Xt,Ys,options);
    [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
    fprintf('\n PD_NBT %.2f%% \n', acc(1));
    time_data = [time_data toc];
    acc_data = [acc_data acc(1)];

end

