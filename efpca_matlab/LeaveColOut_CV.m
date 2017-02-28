function [lam,beta]=LeaveColOut_CV(Ymat,Xmat, VOmega,DOmega, paramstruct)
% choose lam for 
% min |blkdiag(col(Ymat))-blkdiag(col(Xmat))*beta|^2+n*p*lam*beta'*Omega*beta
% using leave-one-column-out CV (essentially, leave-multi-obs-out CV, tend to over-smooth)
% This may be desired in real application (b/c of over-smoothing)
%
% Input: 
%    Ymat       n*p matrix, where Y(:) is the true response vector.
%    Xmat       n*p matrix, where the p columns will be block-diagonalized
%               as a design matrix. In CV, we leave one column of Xmat at a
%               time
%
%    VOmega     p*(p-2) matrix with orthonormal columns
%    DOmega     (p-2)*1 vector of positive values, and
%               Omega=VOmega*diag(DOmega)*VOmega'
%     
%    paramstruct
%          lamrange     a vector of search range for lam. Default is
%                       10.^[-5:0.1:5]. Note: lam is invariant to sample 
%                       size b/c the constant multiplier n*p in front
%          fig          0 or 1, indicator of whether showing the CV plot
%                       default is 0 (no figure). 
%
%
%
% Output:
%   lam         scalar, best tuning parameter based on leave-one-column-out CV
%               when use lam, REMEMBER to multiply by np!
%   beta        p*1 vector, the coefficient estimate corresp to the best lam
%
%
% Need to call: 
%      woodbury.m
%
% By Gen Li, 8/10/2016

[n_,p_]=size(Ymat);
[n,p]=size(Xmat);
% check
if n_~=n || p_~=p
    error('Dimension of Y and X does not match!');
end;

% default values
lamrange=(10.^[-5:0.1:5]); % default searching range
fig=0; % do not show CV score selection figure
if nargin > 4 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'lamrange') ;    
    lamrange = getfield(paramstruct,'lamrange') ; 
    if length(lamrange)==1 % if only one candidate, no need to run this alg
        lam=lamrange;
    return;
    end;
  end ;
  if isfield(paramstruct,'fig') ;    
    fig = getfield(paramstruct,'fig') ; 
  end ;
end;


% calc some useful values
dXTX=diag(Xmat'*Xmat); % p*1
dXTY=diag(Xmat'*Ymat); % p*1
CV_score=zeros(size(lamrange));

for i=1:length(lamrange)
    lambda=n*p*lamrange(i);
    % calculate critical p*p matrix: S=inv(X'X+n*p*lam*Omega), where X is the blkdiag design matrix
    S=woodbury(dXTX,VOmega,lambda*DOmega); % p*p
    beta=S*dXTY; % p*1
    dS=diag(S); %p*1
    Yhat=bsxfun(@times,Xmat,beta');
    
    % calc CV score
    CV_score(i)=( norm(Yhat-Ymat,'fro')^2+sum(  (2*dS-(dS.^2).*dXTX).*(dXTX.*beta-dXTY).^2./(1-dS.*dXTX).^2  ))/p;
    
end;
[~,ind]=min(CV_score);
lam=lamrange(ind);
S=woodbury(dXTX,VOmega,n*p*lam*DOmega); % p*p
beta=S*dXTY; % p*1

if fig
    figure(101);
    plot(log10(lamrange),CV_score);
    xlabel('Log10 of Tuning Parameter','fontsize',15);
    ylabel('Leave-one-column-out CV score','fontsize',15);
    drawnow
end;
