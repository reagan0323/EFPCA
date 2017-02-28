function loss = GrassDist(V1,V2)
% Calculates the grassmannian distance between orthonormal V1 and V2

[p1,r1]=size(V1);
[p2,r2]=size(V2);
if (p1~=p2) 
    error('Input must be matched')
end;

[V1,~,~]=svds(V1,r1);
[V2,~,~]=svds(V2,r2);

loss=norm(acos(svd(V1'*V2)),'fro');

end

   