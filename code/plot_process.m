%% Plot of Nyström Basis Transfer Process
%This script pltos the process of the basis transfer from target to source
%Note for nicer plots the kernelized version of target landmark matrix is
%used.

%% Load Data
close all;
clear all;
fprintf('data=Caltech10_vs_amazon\n');
load('../data/OfficeCaltech/Caltech10_SURF_L10.mat');
fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
Xs = fts;
Ys = labels;

load('../data/OfficeCaltech/amazon_SURF_L10.mat');
fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
Xt = fts;
Yt = labels;


%% Sampling
soureIndx = crossvalind('Kfold', Ys, 2);
targetIndx = crossvalind('Kfold', Yt,2);

Xs = Xs(find(soureIndx==1),:);
Ys = Ys(find(soureIndx==1),:);


Xt = Xt(find(targetIndx==1),:);
Yt = Yt(find(targetIndx==1),:);

%% Pre-Processing
[Xs,Ys] = augmentation(Xs,Xt,Ys);

%% Nyström Basis Transfer
landmarks = 10;
X = full(Xt);
Z = full(Xs);
landmarks =min(min(size(X,1),size(X,2)),landmarks);
idx = [];
C = unique(Ys,'stable');
sizeC = size(C,1);
data = [];
sampleSize = floor( landmarks / sizeC);
class_size = [];

for c = C'
    idxs = find(Ys == c); 
    class_size = [class_size size(idxs,1)];
end
min_size = min(class_size);
sampleSize = min(min_size,sampleSize);




landmarks = sizeC*sampleSize;

%Samples Target
idx = randperm(min(size(X,1),size(X,2)),landmarks);


A = X(idx,:);
B = X(1:landmarks,landmarks+1:end);
F = X(landmarks+1:end,1:landmarks);
C = X(landmarks+1:end,landmarks+1:end);
%Kernelized version for nicer plots
[U,S,H] = svd(A*A');

U_k = [U;F*H(1:size(S,1),:)*pinv(sqrtm(S))];
V_k = [H;B'*U*pinv(sqrtm(S))];

X = U_k*S;

idx = [];
C = unique(Ys,'stable');
sizeC = size(C,1);
data = [];
sampleSize = floor( landmarks / sizeC);
class_size = [];
%Samples Source
for c = C'
    idxs = find(Ys == c); 
    class_size = [class_size size(idxs,1)];
end
min_size = min(class_size);
sampleSize = min(min_size,sampleSize);
for c = C'
    idxs = find(Ys == c); 
    classData= Z(idxs,:);
    y = randsample(size(classData,1),sampleSize);
    classData = classData(y,:);
    data = [data; classData];
end
sampleSize = abs(size(data,1)-landmarks);
c = C(end);
idxs = find(Ys == c);
classData= Z(idxs,:);
y = randsample(size(classData,1),sampleSize);
classData = classData(y,:);
data = [data; classData];


D = diag(svd(data*data',"econ"));
Z = U_k*sqrtm(D);

%% Post-Processing 
Xs = zscore(Z,1);
Xt = zscore(X,1);

%% PLOTTING Data Matrices
i = 1;
for h = {A,data,X,Z,Xt,Xs}


% figure;
% hold on;
h =h{1,1};
s = size(h);
surf(h);
colormap winter
% if mod(i,2)==0
if i==5 ||i==6
    colorbar;
end

set(gcf, 'color', 'none');
if i > 4 
xlabel('Dimensions','FontSize', 12);
ylabel('Samples','FontSize', 12);
end
% hold off;
print(int2str(i),"-dpng","-r1000")
print(int2str(i),"-depsc","-r1000")
i=i+1;
end
% surf(A);