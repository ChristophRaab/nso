function no_dims = MLE_IntrinsicDim(X)
	X = double(unique(X, 'rows'));

    % Make sure data is zero mean, unit variance
    X = X - repmat(mean(X, 1), [size(X, 1) 1]);
    X = X ./ repmat(var(X, 1) + 1e-7, [size(X, 1) 1]);
	% Set neighborhood range to search in
            k1 = 6;  % def 6
            k2 = 12; % def 12
			%fprintf('MLE %d %d \n',k1,k2);

            % Compute matrix of log nearest neighbor distances
            X = X';
            [d n] = size(X);
            X2 = sum(X.^2, 1); 
            knnmatrix = zeros(k2, n);
            if n < 4000
                distance = repmat(X2, n, 1) + repmat(X2', 1, n) - 2 * X' * X;
                distance = sort(distance);
                knnmatrix= .5 * log(distance(2:k2 + 1,:));
            else
                for i=1:n
                    distance = sort(repmat(X2(i), 1, n) + X2 - 2 * X(:,i)' * X);
                    distance = sort(distance);
                    knnmatrix(:,i) = .5 * log(distance(2:k2 + 1))'; 
                end
            end  

            % Compute the ML estimate
            S = cumsum(knnmatrix, 1);
            indexk = repmat((k1:k2)', 1, n);
            dhat = -(indexk - 2) ./ (S(k1:k2,:) - knnmatrix(k1:k2,:) .* indexk);

            % Plot histogram of estimates for all datapoints
            %hist(mean(dhat), 80), pause
            
            % Average over estimates and over values of k
            no_dims = mean(mean(dhat));
            no_dims = round(no_dims);
	end