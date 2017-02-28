# EFPCA
Non-Gaussian Functional Data Analysis
The efpca_matlab folder contains main MATLAB functions for implementing the proposed low-rank exponential family model. It provides a tool box for flexibly analyzing one-way or two-way non-Gaussian functional data. The glmnet_matlab folder is downloaded from http://web.stanford.edu/~hastie/glmnet_matlab/, which contains the main glmnet MATLAB functions. (Note: The glmnet package is not very stable. The MEX files sometimes may crash in MATLAB. In that situation, restarting MATLAB usually solves the problem. All questions regarding the glmnet should be directed to Junyang Qian (junyangq@stanford.edu), who maintains the software.) 

Sim_ModelFitting.m 		a demo showing how to use the main functions in efpca_matlab to estimate the parameters of the proposed model. It contains 5 simulated examples, covering one-way and two-way functional data with Poisson, Bernoulli, and Binomial distributions.

Sim_RankEstimation.m            a demo showing how to use the main functions in efpca_matlab to estimate the latent rank of a non-Gaussian data matrix. 



Main functions in efpca_matlab:

EFPCA_oneway3.m			a function for estimating the latent patterns of one-way functional data with Bernoulli, Poisson, or Gaussian distributions.

EFPCA_twoway3.m			a function for estimating the latent patterns of two-way functional data with Bernoulli, Poisson, or Gaussian distributions.

EFPCA_binomial_twoway3.m	a function for estimating the latent patterns of two-way Binomial functional data.

ExpPCA.m			a function for implementing the exponential PCA (Collins et al, 2001)

LeaveColOut_CV.m		a function for estimating the smoothing parameter in a leave-one-column-out scheme.

LeaveEntOut_CV.m		a function for estimating the smoothing parameter in a leave-one-entry-out scheme.

Nfold_CV_EPCA.m			a function for estimating the latent rank of a non-Gaussian data matrix (for Bernoulli, Binomial, Poisson, and Gaussian)

Param_Bootstrap_EFPCA.m		a function for conducting parametric bootstraps to calculate the point-wise confidence intervals for the estimates. 




Contact: Gen Li, PhD
         Assistant Professor of Biostatistics, Columbia University
         Email: gl2521@columbia.edu  
CopyRight all reserved
Last updated: 2/3/2017
