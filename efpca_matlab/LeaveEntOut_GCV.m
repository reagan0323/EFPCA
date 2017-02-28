function [lam,beta]=LeaveEntOut_GCV(Ymat,Xmat,VOmega,DOmega, paramstruct)
% choose lam for 
% min |blkdiag(col(Ymat))-blkdiag(col(Xmat))*beta|^2+n*p*lam*beta'*Omega*beta
% using leave-one-ENTRY-out CV (to me, the most reasonable CV method, but may under-smooth)
%
% Input: 
%    Ymat       n*p matrix, where Y(:) is the true response vector.
%    Xmat       n*p matrix, where the p columns will be block-diagonalized
%               as a design matrix. 
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

Y=Ymat(:);
diagXTX=diag(Xmat'*Xmat);
XTY=diag(Xmat'*Ymat);

n=size(Y,1);
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

% if only one candidate, no need to perform GCV
if length(lamrange)==1
    lam=lamrange;
    return;
end;
XTX=diag(diagXTX);
CV_score=zeros(size(lamrange));
for i=1:length(lamrange)
    lambda=lamrange(i);
    criticalvalue=woodbury(diagXTX,VOmega,n*lambda*DOmega);
    numerator=(1/n)*(Y'*Y-2*XTY'*criticalvalue*XTY+XTY'*criticalvalue*XTX*criticalvalue*XTY);
    denominator=(1-1/n*trace(XTX*criticalvalue))^2;
 
    CV_score(i)=numerator/denominator;
end;
[~,ind]=min(CV_score);
lam=lamrange(ind);
beta=woodbury(diagXTX,VOmega,n*lam*DOmega)*XTY;

if fig
    figure(101);
    plot(log10(lamrange),CV_score);
    xlabel('Log10 of Tuning Parameter','fontsize',15);
    ylabel('Leave-one-entry-out CV score','fontsize',15);
    drawnow
end;
