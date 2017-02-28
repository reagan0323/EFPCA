function [Omega,VOmega,DOmega]=GetOmega(p)
% get p*p smoothing matrix Omega for smoothing spline (rank p-2)
% VOmega is p*(p-2) left singular matrix
% DOmega is (p-2)*1 singular values
% Note:
% This function only returns Omega for evenly spaced p-dimensional data
% 
Qvv=eye(p)*(-2);
Qvv=spdiags(ones(p,1),1,Qvv);
Qvv=spdiags(ones(p,1),-1,Qvv);
Qvv=Qvv(:,2:(end-1));
Rvv=eye(p-2)*(2/3);
Rvv=spdiags(ones(p-2,1)*(1/6),1,Rvv);
Rvv=spdiags(ones(p-2,1)*(1/6),-1,Rvv);
tempv=chol(inv(Rvv))*Qvv'; % use cholesky decomp to fight against matlab precision of symmetry
Omega=tempv'*tempv; % p*p  singular matrix (rank=p-2)
[VOmega,DOmega,~]=svds(full(Omega),p-2);
DOmega=diag(DOmega);
