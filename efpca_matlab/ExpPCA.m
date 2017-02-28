function [U,V]=ExpPCA(X,r,paramstruct)
% This function implement expoential family PCA (Collins, 2002)
% Alternating GLM (est all layers simultaneously, but no guarantee of 
% orthogonality in each iteration, so we post-process it to be orthogonal)
% Actually, it should be called exponential family SVD rather than PCA,
% because we do not adjust mean for data or constant predictor for natural param.
%
%
% input: 
%
%   X       n*p raw data matrix
%
%   r       scalar, rank of natural parameter matrix 
% 
%   paramstruct  struct('name',value)
%
%       distr   string, distribution name, choose from 
%               'normal', default
%               'poisson',
%               'binomial', need input of N
%               'bernoulli' with binarycure
%       N       n*p matrix of total number of events
%       Tol     stopping criterion of consecutive updates, default 0.1
%       Niter   max number of iterations, default 200
%       binarycure     scalar in [0.8,1], choose an upper quantile for
%                       thresholding binary scores and loadings
%                       An ad hoc approach to address Collins' EPCA failure
%                       on binary data; default=0.95

%
% Output: 
%
%   U       n*r score matrix of natural param matrix, absorbing D, orthogonal 
% 
%   V       p*r loading matrix of natural param matrix, orthonormal
%
%
% Note: this function estimates all ranks of U and V together, rather than 
%       cycling through different components as in Collins
%       For rank=1, no difference; for rank>1, may need to explore the
%       difference
%
% Created on 11/19/2015 by Gen Li
% modified on 2/1/2016 by Gen Li
%         add bernoulli1, a empirical fix (threshold by 95% quantiles) of
%         binary case in Collins algorithm
% modified on 3/1/2016 by Gen Li
%         add bernoulli2, a ridge fix (lambda=0.01) of
%         binary case in Collins algorithm, need access to glmnet

[n,p]=size(X);
if r>n || r>p
    error('Rank exceeds matrix dimension!')
end;

% default parameters
Tol=0.1; % stopping rule
Niter=200; % max iterations
distr='normal';
binarycure=0.95;
N=zeros(n,p);
if nargin > 2 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'Tol') ;    
    Tol = getfield(paramstruct,'Tol') ; 
  end ;
  if isfield(paramstruct,'Niter') ;    
    Niter = getfield(paramstruct,'Niter') ; 
  end ; 
  if isfield(paramstruct,'distr') ;    
    distr = getfield(paramstruct,'distr') ; 
  end ;
  if isfield(paramstruct,'N') ;    
    N = getfield(paramstruct,'N') ; 
  end ;
  if isfield(paramstruct,'binarycure') ;    
    binarycure = getfield(paramstruct,'binarycure') ; 
  end ;
end;



if strcmpi(distr,'normal') % assume variance known as 1
    [U,D,V]=svds(X,r);
    U=U*D;
    
    
elseif strcmpi(distr,'poisson')
    if sum(sum(X<0))
        error('No negative entries for Poisson random variables!')
    end;
    [U,D,V]=svds(log(X+1E-10),r);
    U=U*D;
    U_old=zeros(n,r);
    V_old=zeros(p,r);
    niter=0;
    diff=inf;
    rec=[];
    % alternative updating
    while diff>Tol && niter<Niter
        U_old=U;
        V_old=V;

        % fix U, estimate each row of V
        for j=1:p
            vj = glmfit(U,X(:,j),'poisson','constant','off');
            V(j,:)=vj';
        end;

        % fix V, estimate each row of U
        for i=1:n
            ui = glmfit(V,X(i,:)','poisson','constant','off');
            U(i,:)=ui';
        end;

        % orthogonalize
        [U,D,V]=svds(U*V',r);
        U=U*D;
        asign=sign(U(1,:));
        U=bsxfun(@times,U,asign);
        V=bsxfun(@times,V,asign);
        
        diff=180/pi*acos(min(svd(V'*V_old))); % max principal angle (0~90)
        niter=niter+1;
        % check
        rec=[rec,diff];
%         figure(3);clf;
%         plot(1:niter,rec,'o-');
%         title('Max Principal Angle between two iterations');
%         drawnow;
    end;
    
    if niter==Niter
        disp([distr,' SVD does NOT converge after ',num2str(Niter),' iterations!']);      
    else
        disp([distr,' SVD converges after ',num2str(niter),' iterations.']);      
    end;
    
    
elseif strcmpi(distr,'binomial')
    if sum(sum(N<X))
        error('Binomial total# of events cannot be smaller than success#!')
    end;
    Prob=X./N; % may have NaN entries because N_ij=0;
    temp=rand(n,p)*0.1;
    Prob(Prob==1)=1-temp(Prob==1);
    Prob(Prob==0)=temp(Prob==0);
    Prob(isnan(Prob))=temp(isnan(Prob))*2+0.4;
    Theta=log(Prob./(1-Prob));
    [U,D,V]=svds(Theta,r);
    U=U*D;
    U_old=zeros(n,r);
    V_old=zeros(p,r);
    niter=0;
    diff=inf;
    rec=[];
    % alternative updating
    while diff>Tol && niter<Niter
        U_old=U;
        V_old=V;

        % fix U, estimate each row of V
        for j=1:p
            vj = glmfit(U,[X(:,j),N(:,j)],'binomial','constant','off');
            V(j,:)=vj';
        end;

        % fix V, estimate each row of U
        for i=1:n
            ui = glmfit(V,[X(i,:)',N(i,:)'],'binomial','constant','off');
            U(i,:)=ui';
        end;

        % orthogonalize
        [U,D,V]=svds(U*V',r);
        U=U*D;
        asign=sign(U(1,:));
        U=bsxfun(@times,U,asign);
        V=bsxfun(@times,V,asign);
        
        diff=180/pi*acos(min(svd(V'*V_old))); % max principal angle (0~90)
        niter=niter+1;
        % check
        rec=[rec,diff];
%         figure(3);clf;
%         plot(1:niter,rec,'o-');
%         title('Max Principal Angle between two iterations');
%         drawnow;
    end;
    
    if niter==Niter
        disp([distr,' SVD does NOT converge after ',num2str(Niter),' iterations!']);      
    else
        disp([distr, ' SVD converges after ',num2str(niter),' iterations.']);      
    end;
    
    
% elseif strcmpi(distr,'bernoulli')
%     
%     %%%%%%%%%%%%%%%
%     Niter=20; % override previous setting!!!!! b/c Collins EPCA fails for binary data
%     %%%%%%%%%%%%%%%
%     
%     if sum(sum(X==0 | X==1))~=n*p
%         error('Bernoulli random variable can only take 0 and 1 values!')
%     end;
%     Prob=0.1*rand(n,p)+0.9*(X==1);
%     Theta=log(Prob./(1-Prob));
%     [U,D,V]=svds(Theta,r);
%     U=U*D;
%     U_old=zeros(n,r);
%     V_old=zeros(p,r);
%     niter=0;
%     diff=1;
%     rec=[];
%     % alternative updating
%     while diff>Tol && niter<Niter
%         U_old=U;
%         V_old=V;
% 
%         % fix U, estimate each row of V
%         for j=1:p
%             vj = glmfit(U,X(:,j),'binomial','constant','off');
%             V(j,:)=vj';
%         end;
% 
%         % fix V, estimate each row of U
%         for i=1:n
%             ui = glmfit(V,X(i,:)','binomial','constant','off');
%             U(i,:)=ui';
%         end;
% 
%         % orthogonalize
%         [U,D,V]=svds(U*V',r);
%         U=U*D;
%         asign=sign(U(1,:));
%         U=bsxfun(@times,U,asign);
%         V=bsxfun(@times,V,asign);
%         
%         niter=niter+1;
%         diff=acos(min(svd(V'*V_old))) % max principal angle (0~pi/2)
%         
% %         norm(V-V_old,'fro')
% %         mesh(U*V')
%         % check
%         rec=[rec,diff];
%         figure(3);clf;
%         plot(1:niter,180/pi*rec,'o-');
%         title('Max Principal Angle between two iterations');
%         drawnow;
%     end;  
%     
%     if niter==Niter
%         disp([distr,' SVD does NOT converge after ',num2str(Niter),' iterations!']);      
%     else
%         disp([distr,' SVD converges after ',num2str(niter),' iterations.']);      
%     end;
    
    
% elseif strcmpi(distr,'bernoulli1') % an quantile approach to fix the convergence issue of Collins EPCA for binary data
%     if sum(sum(X==0 | X==1))~=n*p
%         error('Bernoulli random variable can only take 0 and 1 values!')
%     end;
%     Prob=0.1*rand(n,p)+0.9*(X==1);
%     Theta=log(Prob./(1-Prob));
%     [U,D,V]=svds(Theta,r);
%     U=U*D;
%     U_old=zeros(n,r);
%     V_old=zeros(p,r);
%     niter=0;
%     diff=inf;
%     rec=[];
%     % alternative updating
%     while diff>Tol && niter<Niter
%         U_old=U;
%         V_old=V;
% 
%         % fix U, estimate each row of V
%         for j=1:p
%             vj = glmfit(U,X(:,j),'binomial','constant','off');
%             V(j,:)=vj';
%         end;
%         V=bsxfun(@max,bsxfun(@min,V,quantile(V,binarycure,1)),quantile(V,1-binarycure,1)); % this step is to resolve the unbounded likelihood issue with binary data
% 
%         % fix V, estimate each row of U
%         for i=1:n
%             ui = glmfit(V,X(i,:)','binomial','constant','off');
%             U(i,:)=ui';
%         end;
%         U=bsxfun(@max,bsxfun(@min,U,quantile(U,binarycure,1)),quantile(U,1-binarycure,1)); % this step is to resolve the unbounded likelihood issue with binary data
% 
%         % orthogonalize
%         [U,D,V]=svds(U*V',r);
%         U=U*D;
%         asign=sign(U(1,:));
%         U=bsxfun(@times,U,asign);
%         V=bsxfun(@times,V,asign);
%         
%         niter=niter+1;  
%         diff=180/pi*acos(min(svd(V'*V_old))); % max principal angle (0~90)
% 
% %         norm(V-V_old,'fro')
% %         mesh(U*V')
%         % check
%         rec=[rec,diff];
%         figure(3);clf;
%         plot(1:niter,rec,'o-');
%         title('Max Principal Angle between two iterations');
%         drawnow;
%     end;  
%     
%     if niter==Niter
%         disp([distr,' SVD does NOT converge after ',num2str(Niter),' iterations!']);      
%     else
%         disp([distr,' SVD converges after ',num2str(niter),' iterations.']);      
%     end;
% 
%     
    
elseif strcmpi(distr,'bernoulli') % an ridge approach to fix the convergence issue of Collins EPCA for binary data
    if sum(sum(X==0 | X==1))~=n*p
        error('Bernoulli random variable can only take 0 and 1 values!')
    end;
    Prob=0.1*rand(n,p)+0.9*(X==1);
    Theta=log(Prob./(1-Prob));
    [U,D,V]=svds(Theta,r);
    U=U*D;
    U_old=zeros(n,r);
    V_old=zeros(p,r);
    niter=0;
    diff=inf;
    rec=[];
    % alternative updating
    while diff>Tol && niter<Niter
        U_old=U;
        V_old=V;

        % fix U, estimate each row of V
        for j=1:p
            option=struct('alpha',0,'lambda',0.01,'intr',false); 
            fit_v=glmnet(U,X(:,j),'binomial',option);
            V(j,:)=fit_v.beta;
        end;

        % fix V, estimate each row of U
        for i=1:n
            option=struct('alpha',0,'lambda',0.01,'intr',false); 
            fit_u=glmnet(V,X(i,:)','binomial',option);
            U(i,:)=fit_u.beta;
        end;

        % orthogonalize
        [U,D,V]=svds(U*V',r);
        U=U*D;
        asign=sign(U(1,:));
        U=bsxfun(@times,U,asign);
        V=bsxfun(@times,V,asign);
        
        niter=niter+1;  
        diff=180/pi*acos(min(svd(V'*V_old))); % max principal angle (0~90)

%         norm(V-V_old,'fro')
%         mesh(U*V')
%         % check
%         rec=[rec,diff];
%         figure(3);clf;
%         plot(1:niter,rec,'o-');
%         title('Max Principal Angle between two iterations');
%         drawnow;
    end;  
    
    if niter==Niter
        disp([distr,' SVD does NOT converge after ',num2str(Niter),' iterations!']);      
    else
        disp([distr,' SVD converges after ',num2str(niter),' iterations.']);      
    end;


end;
    
    
    
    
    


