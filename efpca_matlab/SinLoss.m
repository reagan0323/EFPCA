function [loss] = SinLoss(a,b)
% Calculates the discrepancy between vectors a and b as measured by
%
% L^2(a,b)= 2|sin^2(a,b)|=\|aa'/|a|^2_2 - bb'/|b|^2_2 \|^2_F

[p1,s1]=size(a);
[p2,s2]=size(b);
if (s1~=1)||(s2~=1) 
    error('Input must be vectors')
end;

if p1~=p2 
    error('Input vectors are of different length')
end;

loss=norm(a*a'/norm(a)^2-b*b'/norm(b)^2,'fro');

end