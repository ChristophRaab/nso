# Nyström Subspace Override
Matlab Source Code for the accepted KI 2020 Paper "Low-Rank Subspace Override for Unsupervised Domain Adaptation". 
[Link to paper (springer)](https://link.springer.com/chapter/10.1007/978-3-030-58285-2_10)

Folders are self-explaining. 
If you encounter any problems with the repository, please open up an issue here or write me a message!

## Demo and Reproducing:
For a demo and reproducing of performance/time results start
_demo.m_

## Main file:
_nso.m_ (Submission Algorithm)

## Secondardy Files:
_pd_ny_svd.m_<br/>
_libsvm (folder)_<br/>
_augmentation.m_
 
## Tutorial:
```matlab 
[model,K,m,n] = nso(Xs',Xt',Ys,options);
[label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
fprintf('\n NSO %.2f%% \n', acc(1));
```
Assume traning data Xs with size d x m and test data Xt with size d x n. Label vector Ys and Yt accordingly. 
nso accecpts the data and an options struct. With this struct the user can specify:
```
NSO Specific:
options.landmarks ~ Number of Landmarks (int)
SVM Specific: 
options.gamma ~ Gamma of SVM (int)
options.smvc ~ C Parameter of SVM (int)
options.ker ~ Kernel Type "linear|rbf|lab" (string)
```
The functions outputs a libsvm model and a kernel over training and test data modified. The training data is modified by NSO algorithm. <br/>
## Reproducing Plots:
Figure 1: Sensitivity of landmark-parameter: _landmarkperformance_plot.m_<br/>
Figure 2: Process of NSO: _plot_process.m_


## Abstract of Submission:
      Domain adaptation focuses on the reuse of supervised learning models in a new context. Prominent applications can be found in robotics, image processing or web mining. In these areas, learning scenarios change by nature, but often remain related and motivate the reuse of existing supervised models.
    While the majority of domain adaptation algorithms utilize all available source and target domain data, we show that efficient domain adaptation requires only a substantially smaller subset from both domains. This makes it more suitable for real-world scenarios where target domain data is rare. The presented approach finds domain invariant representation for source and target data to address domain differences by overriding orthogonal basis structures. By employing a low-rank approximation, the approach remains low in computational time. 
    The presented idea is evaluated on typical domain adaptation tasks with standard benchmark data.
