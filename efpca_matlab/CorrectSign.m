function [newV,ind] = CorrectSign(trueV,V)
% correct column sign in V based on trueV
% trueV is baseline
% V is to be corrected
% no need to to orthogonal
ind=sign(diag(trueV'*V))';
newV=bsxfun(@times,V,ind);
end

   