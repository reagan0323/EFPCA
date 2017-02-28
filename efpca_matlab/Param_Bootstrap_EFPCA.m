function [Uarray,Darray,Varray]=Param_Bootstrap_EFPCA(Utrue,Dtrue,Vtrue,distr,way,paramstruct)
% This is the parametric bootstrap function for obtaining the CI for
% one-way or two-way EFPCA estimates.
%
% input: 
%
%   Utrue       n*r score matrix, with orthonormal columns
%
%   Dtrue       r*1 singular value vector
%
%   Vtrue       p*r loading matrix, with orthonormal columns
%
%   distr   string, choose from 'bernoulli','poisson','normal', 'binomial'
%           if it's binomial, need to specify N in paramstruct
%
%   way     1 or 2, 1 means one-way EFPCA, 2 means 2-way EFPCA
%
%   paramstruct
%
%           N           n*p positive integer matrix, the number of trials, 
%                       specifically for binomial distribution, default=0
%
%           Nboot       scalar, number of bootstraps, 100(default) 
%       
%           CVmethod    1(default): leave-one-entry-out CV
%                       2: leave-one-column-out CV
%
%
% Output: 
%
%   Uarray       n*r*Nboot array, each slice is an estimated U from each bootstrap
% 
%   Darray       r*Nboot matrix, each column is an estimated D from each bootstrap
%
%   Varray       p*r*Nboot array, each slice is an estimated V from each bootstrap
%
%
% Need to call: 
%  CorrectSign.m
%  EFPCA_oneway3.m
%  EFPCA_twoway3.m
%  EFPCA_binomial_twoway3.m
%
% 8.14.2016 by Gen Li

[n,r]=size(Utrue);
[r_1,temp]=size(Dtrue);
[p,r_2]=size(Vtrue);
% check
if r~=r_1 || r~=r_2
    error('Dimensions do not match!')
elseif temp~=1 
    error('Need a column vector of singular values!');
elseif sum(diag(Utrue'*Utrue))>r+0.001 ||  sum(diag(Utrue'*Utrue))<r-0.001 || sum(diag(Vtrue'*Vtrue))>r+0.001 ||  sum(diag(Vtrue'*Vtrue))<r-0.001
    error('U and V need to be normalized!');
elseif way~=1 && way~=2
    error('Can only deal with one-way or two-way smoothness!');
elseif strcmpi(distr,'binomial') &&  ~isfield(paramstruct,'N')
    error('Need to provide N for binomial case!');
end;


% initial values
Nboot=100;
CVmethod=1;
if nargin > 5 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'Nboot') ;    
    Nboot = getfield(paramstruct,'Nboot') ; 
  end ;
  if isfield(paramstruct,'N') ;    
    N = getfield(paramstruct,'N') ; 
  end ;
  if isfield(paramstruct,'CVmethod') ;    
    CVmethod = getfield(paramstruct,'CVmethod') ; 
  end ;
end;


% reproduce key parameters
Theta=Utrue*diag(Dtrue)*Vtrue'; % natural parameter
if strcmpi(distr,'bernoulli') || strcmpi(distr,'binomial')
    param=exp(Theta)./(1+exp(Theta));
elseif strcmpi(distr,'poisson')
    param= exp(Theta);
elseif strcmpi(distr,'normal')
    param=Theta;
end;    


% output
Uarray=zeros(n,r,Nboot);
Darray=zeros(r,Nboot);
Varray=zeros(p,r,Nboot);


%%%%%%%%%%%%%%%%%%%%%
% start bootstrap
%%%%%%%%%%%%%%%%%%%%%
% Note: I code in this way to save computation.
% it may look redundant, but we only need to execute if(way) and if(distr)
% once. 
% In each block, the only difference is X=.... and [U_boot,V_boot]=....

if way==1 % one-way
    if strcmpi(distr,'bernoulli')
        for iboot=1:Nboot
            % simulate random samples from the parameter
            X=binornd(1,param);
            
            % rerun EFPCA
            [U_boot,V_boot]=EFPCA_oneway3(X,distr,r,struct('CVmethod',CVmethod));  
            [U_boot,D_boot,V_boot]=svds(U_boot*V_boot',r);
            D_boot=diag(D_boot);
            [U_boot,ind] = CorrectSign(Utrue,U_boot);
            V_boot=bsxfun(@times,V_boot,ind);
        
            % save data
            Uarray(:,:,iboot)=U_boot;
            Varray(:,:,iboot)=V_boot;
            Darray(:,iboot)=D_boot;
        end;
    elseif strcmpi(distr,'poisson')
        for iboot=1:Nboot
            % simulate random samples from the parameter
            X=poissrnd(param);
            
            % rerun EFPCA
            [U_boot,V_boot]=EFPCA_oneway3(X,distr,r,struct('CVmethod',CVmethod,'U_ini',Utrue*diag(Dtrue),'V_ini',Vtrue));  
            [U_boot,D_boot,V_boot]=svds(U_boot*V_boot',r);
            D_boot=diag(D_boot);
            [U_boot,ind] = CorrectSign(Utrue,U_boot);
            V_boot=bsxfun(@times,V_boot,ind);
        
            % save data
            Uarray(:,:,iboot)=U_boot;
            Varray(:,:,iboot)=V_boot;
            Darray(:,iboot)=D_boot;
        end;
    elseif strcmpi(distr,'normal')
        for iboot=1:Nboot
            % simulate random samples from the parameter
            X=normrnd(param,1);
            
            % rerun EFPCA
            [U_boot,V_boot]=EFPCA_oneway3(X,distr,r,struct('CVmethod',CVmethod));  
            [U_boot,D_boot,V_boot]=svds(U_boot*V_boot',r);
            D_boot=diag(D_boot);
            [U_boot,ind] = CorrectSign(Utrue,U_boot);
            V_boot=bsxfun(@times,V_boot,ind);
        
            % save data
            Uarray(:,:,iboot)=U_boot;
            Varray(:,:,iboot)=V_boot;
            Darray(:,iboot)=D_boot;
        end; 
    else 
        error('No such distribution available for one-way EFPCA.')
    end;
        

elseif way==2
    if strcmpi(distr,'bernoulli')
        for iboot=1:Nboot
            % simulate random samples from the parameter
            X=binornd(1,param);
            
            % rerun EFPCA
            [U_boot,V_boot]=EFPCA_twoway3(X,distr,r,struct('CVmethod',CVmethod));  
            [U_boot,D_boot,V_boot]=svds(U_boot*V_boot',r);
            D_boot=diag(D_boot);
            [U_boot,ind] = CorrectSign(Utrue,U_boot);
            V_boot=bsxfun(@times,V_boot,ind);
        
            % save data
            Uarray(:,:,iboot)=U_boot;
            Varray(:,:,iboot)=V_boot;
            Darray(:,iboot)=D_boot;
        end;
    elseif strcmpi(distr,'poisson')
        for iboot=1:Nboot
            % simulate random samples from the parameter
            X=poissrnd(param);
            
            % rerun EFPCA
            [U_boot,V_boot]=EFPCA_twoway3(X,distr,r,struct('CVmethod',CVmethod));  
            [U_boot,D_boot,V_boot]=svds(U_boot*V_boot',r);
            D_boot=diag(D_boot);
            [U_boot,ind] = CorrectSign(Utrue,U_boot);
            V_boot=bsxfun(@times,V_boot,ind);
        
            % save data
            Uarray(:,:,iboot)=U_boot;
            Varray(:,:,iboot)=V_boot;
            Darray(:,iboot)=D_boot;
        end;
    elseif strcmpi(distr,'normal')
        for iboot=1:Nboot
            % simulate random samples from the parameter
            X=normrnd(param,1);
            
            % rerun EFPCA
            [U_boot,V_boot]=EFPCA_twoway3(X,distr,r,struct('CVmethod',CVmethod));  
            [U_boot,D_boot,V_boot]=svds(U_boot*V_boot',r);
            D_boot=diag(D_boot);
            [U_boot,ind] = CorrectSign(Utrue,U_boot);
            V_boot=bsxfun(@times,V_boot,ind);
        
            % save data
            Uarray(:,:,iboot)=U_boot;
            Varray(:,:,iboot)=V_boot;
            Darray(:,iboot)=D_boot;
        end; 
    elseif strcmpi(distr,'binomial')
        for iboot=1:Nboot
            % simulate random samples from the parameter
            tic
            X=binornd(N,param);
            Tsimdata=toc;
            
            % rerun EFPCA
            tic
            [U_boot,V_boot]=EFPCA_binomial_twoway3(X,N,r,struct('CVmethod',CVmethod));  
            Tefpca=toc;
            [U_boot,D_boot,V_boot]=svds(U_boot*V_boot',r);
            D_boot=diag(D_boot);
            [U_boot,ind] = CorrectSign(Utrue,U_boot);
            V_boot=bsxfun(@times,V_boot,ind);
        
            % save data
            Uarray(:,:,iboot)=U_boot;
            Varray(:,:,iboot)=V_boot;
            Darray(:,iboot)=D_boot;
            save('temp.mat','iboot','Uarray','Varray','Darray');
            disp(['Finish bootstrap ',num2str(iboot),'; time(data, alg)=',...
                num2str(Tsimdata),',',num2str(Tefpca)]);
        end; 
    else 
        error('No such distribution available for two-way EFPCA.')
    end;

end;

