function [Mat]=woodbury(S,V,D)
% this function calculate matrix inverse for 
% T=diag(S)+V*diag(D)*V'
%
% Input: 
%    S      p*1 vector, nonzero values
%
%    V      p*r matrix, no need to be orthogonal
%
%    D      r*1 vector, nonzero values
%
% Output:
%    Mat    p*p matrix, inverse of diag(S)+V*diag(D)*V'
%
% The woodbury identity asserts:
% inv(T)=(S^-1)-(S^-1)*V*inv(D^{-1}+V'*(S^-1)*V)*V'*(S^-1)
% 
% by Gen Li, 3/11/2016

% check dimension
[p,temp1]=size(S);
[r,temp2]=size(D);
[temp_p,temp_r]=size(V);

if temp1~=1 || temp2~=1
    error('Needs to input vectorized diagonal values!');
elseif temp_p~=p || temp_r~=r
    error('Dimension does not match!');
elseif sum(S==0)+sum(D==0)>0
    error('Diagonal values must be non-zero!');
end;


invS=1./S;
invD=1./D;
invSV=bsxfun(@times,V,invS);
Mat=diag(invS)-invSV*inv(diag(invD)+V'*diag(invS)*V)*invSV';
