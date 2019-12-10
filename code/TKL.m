% Domain Invariant Transfer Kernel Learning
% M. Long, J. Wang, J. Sun, and P.S. Yu
% IEEE Transactions on Knowledge and Data Engineering (TKDE)

% Contact: Mingsheng Long (longmingsheng@gmail.com)

function K = TKL(Xs, Xt, options)

if nargin < 3
    error('Algorithm parameters should be set!');
end
if ~isfield(options, 'ker')
    options.ker = 'linear';
end
if ~isfield(options, 'gamma')
    options.gamma = 1.0;
end
if ~isfield(options, 'eta')
    options.eta = 2.0;
end
ker = options.ker;
gamma = options.gamma;
eta = options.eta;

%fprintf('TKL: ker=%s  gamma=%f  eta=%f\n', ker, gamma, eta);

X = [Xs, Xt];
m = size(Xs, 2);
K = kernel('rbf',X,[],gamma);
K = K + 1e-6 * eye(size(K,2));
Ks = K(1:m, 1:m);
Kt = K(m+1:end, m+1:end);
Kst = K(1:m, m+1:end);
Kat = K(:, m+1:end);

dim = min(size(Kt, 2) - 10, 200);
[Phit,Lamt] = eigs((Kt + Kt') / 2, dim, 'LM');
Phis = Kst * Phit * Lamt^(-1);
Phia = Kat * Phit * Lamt^(-1);

A = Phis' * Phis;
B = Phis' * Ks * Phis;
Q = A .* A;
Q = (Q + Q') / 2;
r = -diag(B);
Anq = diag(-ones(dim, 1)) + diag(eta .* ones(dim - 1, 1), 1);
bnq = zeros(dim, 1);
lb = zeros(dim, 1);
optim = optimset('Algorithm', 'interior-point-convex', 'MaxIter', 1000, 'TolFun', 1e-16, 'TolX', 1e-16, 'Display', 'off');
lambda = quadprog(Q, r, Anq, bnq, [], [], lb, [], [], optim);

K = Phia * diag(lambda) * Phia';
K = (K + K') / 2;

% fprintf('TKL: terminated!\n');

end
