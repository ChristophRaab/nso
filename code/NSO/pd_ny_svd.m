function [X,Z] = pd_ny_svd(X,Z,Ys,landmarks)
%NY_SVD Summary of this function goes here
%   Detailed explanation goes here
X = full(X);
Z = full(Z);
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


A = X(idx,idx);
B = X(1:landmarks,landmarks+1:end);
F = X(landmarks+1:end,1:landmarks);
C = X(landmarks+1:end,landmarks+1:end);

[U,S,H] = svd(A,"econ");

U_k = [U;F*H*pinv(S)];
V_k = [H;B'*U*pinv(S)];

X = (U_k/norm(U_k))*S;



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


D = diag(svd(data,"econ"));
Z = (U_k/norm(U_k))*D;
end

