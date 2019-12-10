function [X,Z] = ny_svd(X,Z,landmarks)
%NY_SVD Summary of this function goes here
%   Detailed explanation goes here
X = full(X);
Z = full(Z);
landmarks =min(min(size(X,1),size(X,2)),landmarks);
idx = randperm(min(size(X,1),size(X,2)),landmarks);

A = X(idx,idx);
B = X(1:landmarks,landmarks+1:end);
F = X(landmarks+1:end,1:landmarks);
C = X(landmarks+1:end,landmarks+1:end);

[U,S,H] = svd(A);

U_k = [U;F*H*pinv(S)];
V_k = [H;B'*U*pinv(S)];

X = U_k*S;

A = Z(idx,idx);
[L,D,R] = svd(A);
Z = U_k*D;

end

