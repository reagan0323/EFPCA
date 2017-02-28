function [avgCVscore,rOpt,allCVscore]=Nfold_CV_EPCA(X,distr,rcand,Nfold,paramstruct)
% This function calc N-fold CV scores for an EPCA problem for a set of ranks 
%
% Input
%     X         n*p fully observed data matrix, from exponential family
%               or 1*2 cell array for binomial distribution {NumSuccess,NumTrial}
%     distr     string, specifying distribution
%               'normal','poisson','bernoulli','binomial'
%     rcand     a vector of candidate ranks for natural parameter matrix
%     Nfold     number of fold, at this point, need to be a divisor of n*p
%               suggest 4~10
%
%     paramstruct
%          seed    scalar, the random split seed, default is a fixed number
% 
% Output
%     avgCVscore    a vector (same size with rcand) of CV scores, 
%                   each entry is avg (x-hat{x})^2 across folds
%     rOpt          the optimal rank with the smallest CV score in rcand
%
%     allCVscore    Nfold*length(rcand) matrix, contains all CV scores, each row corresponds to a
%                   fold and each col corresponds to a tuning param
%
% by Gen Li, 3/30/2016
%      basically ready for prime!

seed=20160529;
if nargin > 4 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'seed') ;    
    seed = getfield(paramstruct,'seed') ; 
  end ;
end;
rng(seed);% set seed


if strcmpi(distr,'binomial')
    N=X{2}; % total number of trials    
    X=X{1}; % total number of success
end;
disp(['Leave out ',num2str(100/Nfold),'% entries for EPCA CV...']);

% split data into Nfold 
[n,p]=size(X);
% check
if n*p/Nfold-floor(n*p/Nfold)~=0
    error('Number of CV folds need to be a divisor of n*p...');
end
blacklist=reshape(randsample(n*p,n*p),n*p/Nfold,Nfold); % each column corresp to the index of leftout samples in each fold, nonoverlap
% deal with special input with binomial



CVscore=[];
ifold=1;
while ifold<=Nfold % ntimes cross validation 
    disp(['Fold ',num2str(ifold)]);
    % creat sparse data for this fold
    ind = blacklist(:,ifold);
    MisInd=zeros(n,p);
    MisInd(ind)=1; % missing index, 1=missing

    if sum(sum(MisInd,1)==n)>0 || sum(sum(MisInd,2)==p)>0
        warning('This fold contains missing rows or columns...skip...');
        ifold=ifold+1;
        continue
    else
        if strcmpi(distr,'binomial')
            [score,~]=CV_binomialPCA(N,X,rcand,MisInd);
        else
            [score,~]=CV_EPCA(X,distr,rcand,MisInd);
        end;
        CVscore=[CVscore;score];
        ifold=ifold+1;
    end;
end;
allCVscore=CVscore;
avgCVscore=mean(CVscore,1); % this avg is ok
[~,ind]=min(avgCVscore);
rOpt=rcand(ind);

end










function [CVscore,total_mis]=CV_EPCA(X,distr,rcand,MisInd)
% This function calc CV scores for an EPCA problem for a set of ranks 
% in one realization of multiple missing observations (indicated by MisInd). 
%
% Input
%     X         n*p fully observed data matrix, from exponential family
%     distr     string, specifying distribution
%     rcand     a vector of candidate ranks for natural parameter matrix
%     MisInd    n*p 0/1 matrix, corresponding to X, 1=missing
%
% Output
%     CVscore       a vector of CV scores corresp to rcand, each entry is 
%                   sum (x_i-hat{x_i})^2 /N_i, where i is the index of MisInd=1
%     total_mis     total number of missing entries
%
%
% by Gen Li, 3/21/2016
% Modified on 3/30/2016
%          change bernoulli name to bernoulli
%          use distr-specific metric functions to calc CV score
% Modified on 4/1/2016
%          use cheated initial value for poisson
%          use glmfit instead of glmnet for poisson (to avoid crash)
%



[n,p]=size(X);
[n_,p_]=size(MisInd);
if n_~=n || p_~=p
    error('Missing Index matrix and Data matrix not compatible!');
end;
CVscore=zeros(size(rcand));
total_mis=sum(sum(MisInd));

% specific exponential family functions
if strcmpi(distr,'normal')
    distr1='gaussian'; % rename for glmnet
    fcn_db=@(theta)(theta);
elseif strcmpi(distr,'bernoulli')
    distr1='binomial';
    fcn_db=@(theta)exp(theta)./(1+exp(theta));
elseif strcmpi(distr,'poisson')
    distr1='poisson';
    fcn_db=@(theta)exp(theta);
end;   


% run for different ranks
for irun=1:length(rcand)
    r=rcand(irun);
    disp(['Running Rank ',num2str(r),':']);
    
    % initial value for EPCA
    if strcmpi(distr,'poisson') % cheat a bit in initial values
        [U,D,V]=svds(log(X+1E-5),r);
        U=U*D;
    else
        U=randn(n,r);
        V=randn(p,r);
    end;

    % perform EPCA with missing entries
    diff=inf;
    niter=0;
    Niter=100;
    % rec=[];
    while diff>0.1 && niter<Niter
        V_old=V;
        % fix U, estimate each row of V
        for j=1:p
            ValidEntry=find(MisInd(:,j)==0);
            currY=X(ValidEntry,j);
            currX=U(ValidEntry,:);
            if strcmpi(distr,'poisson')  % b/c glmnet unstable for spiked poisson
                tempv=glmfit(currX,currY,'poisson','constant','off');
                V(j,:)=tempv;
            elseif strcmpi(distr,'normal')
                V(j,:)=inv(currX'*currX)*currX'*currY;
            else
%                 tempv=glmfit(currX,currY,'poisson','constant','off');
%                 V(j,:)=tempv;
                option=struct('alpha',0,'lambda',0.05,'intr',false);  % use small ridge
                fit_v=glmnet(currX,currY,distr1,option);
                V(j,:)=fit_v.beta;
            end;
        end;
        % fix V, estimate each row of U
        for i=1:n
            ValidEntry=find(MisInd(i,:)==0);
            currY=X(i,ValidEntry)';
            currX=V(ValidEntry,:);
            if strcmpi(distr,'poisson') 
                tempu=glmfit(currX,currY,'poisson','constant','off');
                U(i,:)=tempu;
            elseif strcmpi(distr,'normal')
                U(i,:)=inv(currX'*currX)*currX'*currY;
            else
%                 tempu=glmfit(currX,currY,'poisson','constant','off');
%                 U(i,:)=tempu;
                option=struct('alpha',0,'lambda',0.05,'intr',false); 
                fit_u=glmnet(currX,currY,distr1,option);
                U(i,:)=fit_u.beta;
            end;
        end;
        % orthogonalize
        [U,D,V]=svds(U*V',r);
        U=U*D;
        asign=sign(U(1,:));
        U=bsxfun(@times,U,asign);
        V=bsxfun(@times,V,asign);

        diff=180/pi*acos(min(svd(V'*V_old))); % max principal angle (0~90)
        niter=niter+1;

    end;

    if niter==Niter
        disp(['NOT converge after ',num2str(Niter),' iterations! Final PrinAngle=',num2str(diff)]);      
    else
        disp(['Converge after ',num2str(niter),' iterations.']);      
    end;


    % refit missing values
    Theta=U*V';
    Mu=fcn_db(Theta);
    Xtrue=X(MisInd==1);
    Xpred=Mu(MisInd==1);
    CVscore(irun)=ExpMetric(Xtrue,Xpred,distr);
end;

end






function [CVscore,total_mis]=CV_binomialPCA(N,X,rcand,MisInd)
% This function calc CV scores for truly binomial 
% we separate this from CV_EPCA because glmnet cannot handle truly binomial  
% by Gen Li, 3/30/2016
% Note: this may not be necessary since glmnet CAN handle binomial (6/12/2016)

[n,p]=size(X);
[n_,p_]=size(MisInd);
if n_~=n || p_~=p
    error('Missing Index matrix and Data matrix not compatible!');
end;
CVscore=zeros(size(rcand));
total_mis=sum(sum(MisInd));

% specify b'(theta) for binomial
fcn_db=@(theta)N.*exp(theta)./(1+exp(theta));


% run for different ranks
for irun=1:length(rcand)
    r=rcand(irun);
    disp(['Running Rank ',num2str(r),':']);
    
    % initial value for EPCA 
    %(cheat a little bit here, by assuming no missing value)
    tempp=X./N;
    [U,D,V]=svds(log(tempp./(1-tempp)),r);
    U=U*D;

    % perform EPCA with missing entries
    diff=inf;
    niter=0;
    Niter=100;
    % rec=[];
    while diff>0.1 && niter<Niter
        V_old=V;
        % fix U, estimate each row of V
        for j=1:p
            ValidEntry=find(MisInd(:,j)==0);
            currY=[N(ValidEntry,j)-X(ValidEntry,j),X(ValidEntry,j)];
            currX=U(ValidEntry,:);
            option=struct('alpha',0,'lambda',0,'intr',false);  % no penalty
            fit_v=glmnet(currX,currY,'binomial',option);
            V(j,:)=fit_v.beta;
%             ValidEntry=find(MisInd(:,j)==0);
%             currY=[X(ValidEntry,j),N(ValidEntry,j)];
%             currX=U(ValidEntry,:);
%             tempv=glmfit(currX,currY,'binomial','constant','off');
%             V(j,:)=tempv;
        end;
        % fix V, estimate each row of U
        for i=1:n
            ValidEntry=find(MisInd(i,:)==0);
            currY=[N(i,ValidEntry)'-X(i,ValidEntry)',X(i,ValidEntry)'];
            currX=V(ValidEntry,:);
            option=struct('alpha',0,'lambda',0,'intr',false); 
            fit_u=glmnet(currX,currY,'binomial',option);
            U(i,:)=fit_u.beta;
%             ValidEntry=find(MisInd(i,:)==0);
%             currY=[X(i,ValidEntry)',N(i,ValidEntry)'];
%             currX=V(ValidEntry,:);
%             tempu=glmfit(currX,currY,'binomial','constant','off');
%             U(i,:)=tempu;
        end;
        % orthogonalize
        [U,D,V]=svds(U*V',r);
        U=U*D;
        asign=sign(U(1,:));
        U=bsxfun(@times,U,asign);
        V=bsxfun(@times,V,asign);

        diff=180/pi*acos(min(svd(V'*V_old))); % max principal angle (0~90)
        niter=niter+1;
    end;

    if niter==Niter
        disp(['NOT converge after ',num2str(Niter),' iterations! Final PrinAngle=',num2str(diff)]);      
    else
        disp(['Converge after ',num2str(niter),' iterations.']);      
    end;


    % refit missing values
    Theta=U*V';
    Mu=fcn_db(Theta);
    Xtrue=X(MisInd==1);
    Xpred=Mu(MisInd==1);
    Ntrue=N(MisInd==1);
%     CVscore(irun)=mean(abs(Xtrue-Xpred)./sqrt(Xpred.*(Ntrue-Xpred)./Ntrue));
    CVscore(irun)=sum(abs(Xtrue-Xpred)./sqrt(Xpred.*(Ntrue-Xpred)./Ntrue)); % sum of abs pearson residuals
end;

end




function dev=ExpMetric(Xtrue,Xpred,distr)
% this function calc the deviance of two vectors from specified distribution
% Can accommodate normal, poisson, and bernoulli
% The deviance function for binomial is embedded in CV_binomialPCA
n1=length(Xtrue(:));
n2=length(Xpred(:));
Xtrue=Xtrue(:);
Xpred=Xpred(:);

if n1~=n2
    error('Not equal size...');
end;

if strcmpi(distr,'normal') || strcmpi(distr,'gaussian')
%     dev=norm(Xtrue-Xpred,'fro')^2/n1;
    dev=sum(abs(Xtrue-Xpred));
elseif strcmpi(distr,'poisson')
%     dev=mean(abs(Xtrue-Xpred)./sqrt(Xpred));
    dev=sum(abs(Xtrue-Xpred)./sqrt(Xpred));
elseif strcmpi(distr,'bernoulli')
%     dev=mean(abs(Xtrue-Xpred)./sqrt(Xpred.*(1-Xpred)));
    dev=sum(abs(Xtrue-Xpred)./sqrt(Xpred.*(1-Xpred)));
else
    error('No such distribution...');
end;


end