function [prediction ,accuracy] = my_kernel_knn(M, Xr, Yr, Xt, Yt)
% ref: Geodesic Flow Kernel for Unsupervised Domain Adaptation.  
% B. Gong, Y. Shi, F. Sha, and K. Grauman.  
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, RI, June 2012.

dist = repmat(diag(Xr*M*Xr'),1,length(Yt)) + repmat(diag(Xt*M*Xt')',length(Yr),1) - 2*Xr*M*Xt';
[~, minIDX] = min(dist);
prediction = Yr(minIDX);
accuracy = sum( prediction==Yt ) / length(Yt); 
accuracy = full(accuracy);
end