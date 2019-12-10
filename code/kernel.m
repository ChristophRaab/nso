%% Kernel function by Mingsheng Long
% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','lap'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013

function K = kernel(ker, X, X2, gamma)

if ~exist('ker', 'var')
    ker = 'linear';
end
if ~exist('gamma', 'var')
    gamma = 1.0;
end

switch ker
    case 'linear'
        if isempty(X2)
            K = X' * X;
        else
            K = X' * X2;
        end
        
    case 'rbf'
        n1sq = sum(X.^2, 1);
        n1 = size(X, 2);
        if isempty(X2)
            D = (ones(n1, 1) * n1sq)' + ones(n1, 1) * n1sq -2 * (X' * X);
        else
            n2sq = sum(X2.^2, 1);
            n2 = size(X2, 2);
            D = (ones(n2, 1) * n1sq)' + ones(n1, 1) * n2sq -2 * X' * X2;
        end
        gamma = gamma / mean(mean(D));
        K = exp(-gamma * D);
        
    case 'lap'
        n1sq = sum(X.^2, 1);
        n1 = size(X, 2);
        if isempty(X2)
            D = (ones(n1, 1) * n1sq)' + ones(n1, 1) * n1sq -2 * (X' * X);
        else
            n2sq = sum(X2.^2, 1);
            n2 = size(X2, 2);
            D = (ones(n2, 1) * n1sq)' + ones(n1, 1) * n2sq -2 * X' * X2;
        end
        gamma = gamma / mean(mean(D));
        K = exp(-sqrt(gamma * D));
        
    otherwise
        error(['Unsupported kernel ' ker])
end

if size(K, 1) == size(K, 2)
    K = (K + K') / 2;
end

end
