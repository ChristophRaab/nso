% Copyright (c) 2013, Basura Fernando
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification, 
% are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this 
%list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation and/or 
% other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Based on the paper :
% 
% @inproceedings{Fernando2013b,
% author = {Basura Fernando, Amaury Habrard, Marc Sebban, Tinne Tuytelaars},
% title = {Unsupervised Visual Domain Adaptation Using Subspace Alignment},
% booktitle = {ICCV},
% year = {2013},
% } 
%

% Inputs 
% Source_Data : normalized source data. Use NormalizeData function to
% normalize the data before
%
% Target_Data :normalized target data. Use NormalizeData function to
% normalize the data before
% 
% Xs : source eigenvectors obtained from normalized source data (e.g. PCA)
% Xt : target eigenvectors obtained from normalized source data (e.g. PCA)
% 
% Source_label : source class label
% Target_label : target class label
%
%
function [predicted_label, accuracy, decision_values,model] =  Subspace_Alignment(Source_Data,Target_Data,Source_label,Target_label,Xs,Xt)

% Subspace alignment and projections
Target_Aligned_Source_Data = Source_Data*(Xs * Xs'*Xt);
Target_Projected_Data = Target_Data*Xt;

A = (Xs*Xs')*(Xt*Xt');
Sim = Source_Data * A *  Target_Data';
[predicted_label, accuracy, decision_values,model] = SVM_Accuracy (Source_Data, A,Target_label,Sim,Source_label);



end
% 
% function Data = NormalizeData(Data)
%     Data = Data ./ repmat(sum(Data,2),1,size(Data,2)); 
%     Data = zscore(Data,1);  
% end


function [predicted_label, accuracy, decision_values,model] = SVM_Accuracy (trainset, M,testlabelsref,Sim,trainlabels)
	Sim_Trn = trainset * M *  trainset';
	index = [1:1:size(Sim,1)]';
	Sim = [[1:1:size(Sim,2)]' Sim'];
	Sim_Trn = [index Sim_Trn ];    
	
	 
	 K = kernel('rbf', [Sim_Trn',Sim'], [],1);
     m = size(Sim_Trn, 1);
     n = size(Sim, 1);

     model = svmtrain(full(trainlabels), [(1:m)', K(1:m, 1:m)], ['-c 10', ' -t 4 -q 1']);
     [predicted_label, accuracy, decision_values] = svmpredict(full(testlabelsref), [(1:n)', K(m+1:end, 1:m)], model);
end


% function acc = LinAccuracy(trainset,testset,trainlbl,testlbl)	           
% 		model = trainSVM_Model(trainset,trainlbl);
%         [predicted_label, accuracy, decision_values] = svmpredict(testlbl, testset, model);
%         acc = accuracy(1,1);	
% end
% 
% function svmmodel = trainSVM_Model(trainset,trainlbl)
%     C = [0.001 0.01 0.1 1.0 10 100 ];   
%     parfor i = 1 :size(C,2)
%         model(i) = svmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q -v 2',C(i) )); 
%     end
%     [val indx]=max(model); 
%     CVal = C(indx);
%     svmmodel = svmtrain(double(trainlbl), sparse(double((trainset))),sprintf('-c %d -q',CVal));
% end