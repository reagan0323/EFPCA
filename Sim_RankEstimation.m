% This file validates the CV approach for estimating the latent rank of a
% non-Gaussian data matrix. We consider three different settings: 
% bernoulli, poisson, and normal
%
% Contact: Gen Li, PhD
%          Assistant Professor of Biostatistics, Columbia University
%          Email: gl2521@columbia.edu  
%
% CopyRight all reserved
% Last updated: 2/3/2017


%% simulate data
addpath efpca_matlab
addpath glmnet_matlab
n=100;
p=50;
r=5; 
rng(20160612);
[U,D,V]=svds(randn(n,p),r);
U=max(min(U,quantile(U(:),0.80)),-quantile(U(:),0.80));
V=max(min(V,quantile(V(:),0.80)),-quantile(V(:),0.80));
U=GramSchmidt(U);
V=GramSchmidt(V);

choosesetting=1;
switch choosesetting
    case 1 % bernoulli 
        distr='bernoulli';
        Theta=U*diag([120,100,90,80,60])*V';
        param= exp(Theta)./(1+exp(Theta));
        X=binornd(1,param);
        figure();hist(param(:)) 
    case 2 % poisson 
        distr='poisson';
        Theta=U*diag([25,25,25,20,20])*V';
        param= exp(Theta); 
        X=poissrnd(param);
        figure();mesh(param)     
        N=zeros(n,p);
    case 3 % normal
        distr='normal';
        Theta=U*diag([45,40,35,30,25])*V';
        param= Theta; 
        X=normrnd(param,1);
        figure();mesh(param)     
        N=zeros(n,p);        
end;

%% run CV
rcand=1:10;
Nfold=10;
[avgCVscore,rOpt,allCVscore]=Nfold_CV_EPCA(X,distr,rcand,Nfold);

% plot CV scores for different folds
figure(1);clf;
plot(allCVscore','k*--','linewidth',1);
hold on;
plot(avgCVscore,'ro-','linewidth',2);
plot(rcand,ones(size(rcand))*min(avgCVscore),'r:','linewidth',1.5)
set(gca,'fontsize',20);
xlim([1,9]);
ylim([400,1000]);
xlabel('Ranks','fontsize',25);
ylabel('Cross Validation Score','fontsize',25);
title('Poisson Data CV Rank Estimation','fontsize',30);
