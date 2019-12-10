%% Landmark Performance Plot

% %% Image Plot
% result = [];
% testsize = 3;
% options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
% options.gamma = 1.0;         % TKL: width of gaussian kernel
% landmarks = [10,50,100,150,200,300,400,500,600,700,1000,1100,1500];
% result = [];
% srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
% tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
% 
% for l = landmarks
%     options.landmarks = l;
%     errors = [];
%     for iData = 1:12
%         for i = 1:testsize
%             src = char(srcStr{iData});
%             tgt = char(tgtStr{iData});
%             data = strcat(src, '_vs_', tgt);
%             fprintf('data=%s\n', data);
%             load(['../data/OfficeCaltech/' src '_SURF_L10.mat']);
%             fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
%             Xs = zscore(fts, 1);
%             Ys = labels;
%             
%             load(['../data/OfficeCaltech/' tgt '_SURF_L10.mat']);
%             fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
%             Xt = zscore(fts, 1);
%             Yt = labels;
%             %  Xs-SCORE and Sampling
%             soureIndx = crossvalind('Kfold', Ys, 2);
%             targetIndx = crossvalind('Kfold', Yt,2);
%             Xs = Xs(find(soureIndx==1),:);
%             Ys = Ys(find(soureIndx==1),:);
%             
%             Xt = Xt(find(targetIndx==1),:);
%             Yt = Yt(find(targetIndx==1),:);
%             m = size(Xs, 1);
%             n = size(Xt, 1);
%             
%             %% NBT
%             [model,K,m,n] = pd_cl_nbt(Xs,Xt,Ys,options);
%             [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
%             fprintf('\n Error: PD_NBT %.2f%% \n', 100-acc(1));
%             errors = [errors; 100-acc(1)];
%         end
%     end
%     result = [result errors];
% end
% result = [landmarks; result];
% save(strcat("image_landmark_performance.mat"),'result');
% load image_landmark_performance.mat
% 
% %%
% result = result';
% x = result(1:13,1);
% v = mean(result(1:13,2:end)');
% xq = [1:700,0.1];
% figure
% vq1 = interp1(x,v,xq,'spline');
% vq1 = vq1(1:end-1);
% xq = xq(1:end-1);
% plot(xq,vq1,'LineWidth',6,'color',[0.8500, 0.3250, 0.0980]);
% xlabel('Number of Landmarks','FontSize', 22);
% ylabel('Classification Error','FontSize', 2);
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset;
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
% print("image_error_vs_landmarks","-depsc","-r3000")
% 
% %% Test Result Plot
% result = [];
% testsize = 3;
% options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
% options.gamma = 1.0;         % TKL: width of gaussian kernel
% landmarks = [10,50,100,150,200,300,400,500,600,700,1000,1100,1500];
% result = [];
% for l = landmarks
%     options.landmarks = l;
%     errors = [];
%     for name =  {'org_vs_people','org_vs_place', 'people_vs_place'}
%         for iData = 1:2
% 
%             for i = 1:testsize
%                 data = char(name);
%                 data = strcat(data, '_', num2str(iData));
%                 load(strcat('../data/Reuters/', data));
% 
%                 fprintf('data=%s\n', data);
% 
%                 Xs =full(Xs);
%                 Xt = full(Xt);
%                 Xs = zscore(Xs,1);
%                 Xt = zscore(Xt,1);
%                 %  Xs-SCORE and Sampling
%                 Xs = Xs';Xt = Xt';
%                 soureIndx = crossvalind('Kfold', Ys, 2);
%                 targetIndx = crossvalind('Kfold', Yt,2);
%                 Xs = Xs(find(soureIndx==1),:);
%                 Ys = Ys(find(soureIndx==1),:);
% 
% 
%                 Xt = Xt(find(targetIndx==1),:);
%                 Yt = Yt(find(targetIndx==1),:);
%                 m = size(Xs, 2);
%                 n = size(Xt, 2);
% 
%                 %% NBT
%                 [model,K,m,n] = pd_cl_nbt(Xs,Xt,Ys,options);
%                 [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
%                 fprintf('\n Error: PD_NBT %.2f%% \n', 100-acc(1));
%                 errors = [errors; 100-acc(1)];
%             end
%         end
%     end
%     result = [result errors];
% end
% result = [landmarks; result];
% save(strcat("text_landmark_performance.mat"),'result');
% load text_landmark_performance.mat
% result = result';
% x = result(:,1);
% v = mean(result(:,2:end)');
% xq = [1:1000,0.1];
% figure
% vq1 = interp1(x,v,xq,'spline');
% vq1 = vq1(1:end-1);
% xq = xq(1:end-1);
% 
% plot(xq,vq1,'LineWidth',6,'color',[0.8500, 0.3250, 0.0980]);
% xlabel('Number of Landmarks','FontSize', 22);
% ylabel('Classification Error','FontSize', 22);
% ax = gca;
% outerpos = ax.OuterPosition;
% ti = ax.TightInset;
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% ax.Position = [left bottom ax_width ax_height];
% print("text_error_vs_landmarks","-depsc","-r3000")
% 
srcStr = {'Caltech', 'Caltech', 'Caltech', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech', 'webcam', 'dslr', 'Caltech', 'amazon', 'dslr', 'Caltech', 'amazon', 'webcam'};
testsize =10;
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.gamma = 1.0;         % TKL: width of gaussian kernel
accuracy = [];
stdd = [];
time = [];
landmarks = [10,50,100,150,200,250,300,500,700,1000,2000,3000,4000]
for l = landmarks
    options.landmarks = l;
    errors = [];
    for iData = 1:12

        for i = 1:testsize
            
        src = char(srcStr{iData});
        tgt = char(tgtStr{iData});
        data = strcat(src, '_vs_', tgt);
        fprintf('data=%s\n', data);
        load(['../datasets/domain_adaptation/OfficeCaltech/features/Decaf/' src '_decaf.mat']);
        feas = feas ./ repmat(sum(feas, 2), 1, size(feas, 2));
        Xs = zscore(feas, 1);
        Ys = labels;
        
        load(['../datasets/domain_adaptation/OfficeCaltech/features/Decaf/' tgt '_decaf.mat']);
        feas = feas ./ repmat(sum(feas, 2), 1, size(feas,2));
        Xt = zscore(feas, 1);
        Yt = labels;
    
        Xs = zscore(Xs);
        Xt = zscore(Xt);
  
        %% NBT
        options.landmarks = l;
        tic;
        [model,K,m,n] = pd_cl_nbt(Xs,Xt,Ys,options);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\n PD_NBT %.2f%% \n', acc(1));
 
                errors = [errors; 100-acc(1)];
        end
    end
    result = [result errors];
end
result = [landmarks 100-accuracy];
save(strcat("decaf_oc_landamarks.mat"),'result');
load decaf_oc_landamarks.mat
result = result';
x = result(:,1);
v = mean(result(:,2:end)');
xq = [1:1000,0.1];
figure
vq1 = interp1(x,v,xq,'spline');
vq1 = vq1(1:end-1);
xq = xq(1:end-1);

plot(xq,vq1,'LineWidth',6,'color',[0.8500, 0.3250, 0.0980]);
xlabel('Number of Landmarks','FontSize', 22);
ylabel('Classification Error','FontSize', 22);
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
print("text_error_vs_landmarks","-depsc","-r3000")
