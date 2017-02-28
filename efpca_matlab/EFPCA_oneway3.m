function [U,V]=EFPCA_oneway3(data,distr, r, paramstruct)
% This is the function for one-way EFPCA for normal, binary, and poisson.
% It estimates multi-rank natural parameter matrix decomposition.
% Loadings are smooth (with different degrees of smoothness)
%
%
% input: 
%
%   data    n*p data matrix from some exponential family distribution
%
%   distr   string, specify the distribution, 
%            choose from 'bernoulli','poisson','normal'
%            ( not available for 'binomial' yet)
%
%   r       scalar, prespecified rank
%
%   paramstruct
%
%           Tol         converging threshold for max principal angle
%                       between consecutive V estimates, default = 0.1 degree
%
%           Tol_inner   converging threshold for inner iterations of IRLS
%                       for estimate of each loading, default = 0.1 degree
%
%           Niter       max number of iteration through multi-ranks,
%                       default=200
% 
%           niter_lam   number of outer iterations for floating tuning parameters
%                       After niter_lam, all 2r tuning parameters will be
%                       fixed. Default is 20
%
%           U_ini       n*r initial estimate
%
%           V_ini       p*r initial estimate
%          
%           Omegav      p*p matrix for roughtness penalty, default is classic
%                       smoothing spline Omega matrix
%
%           CVmethod    scalar, 1(default) is leave-one-entry-out CV
%                       2 is leave-one-column-out CV (smoother estimates)
%
% Output: 
%
%   U       n*r smooth score vector, orthogonal columns
% 
%   V       p*r smooth loading matrix, orthonormal columns
%
% Need to call: 
%  PinAngle.m
%  GetOmega.m
%  LeaveColOut_CV.m or LeaveEntOut_GCV.m
%  glmnet.m
%
% 8/14/2016 by Gen Li 

[n,p]=size(data);
tdata=data';
if r>min(n,p)
    error('Rank too large!')
end;

% initial values
Tol=0.1; % overall threshold
Tol_inner=0.1; % threshold for each component 
Niter=200;
Niter_inner=10; % max number of iterations for each penalized IRLS problem
niter_lam=20; % max number of outer iterations with floating lambdas
CVmethod=1; % leave-one-entry out CV
V=max(-.5,min(randn(p,r),.5));
U=max(-.5,min(randn(n,r),.5));
[~,VOmegav,DOmegav]=GetOmega(p);
%
if nargin > 3 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'Tol') ;    
    Tol = getfield(paramstruct,'Tol') ; 
  end ;
  if isfield(paramstruct,'Tol_inner') ;    
    Tol_inner = getfield(paramstruct,'Tol_inner') ; 
  end ;
  if isfield(paramstruct,'Niter') ;    
    Niter = getfield(paramstruct,'Niter') ; 
  end ;
  if isfield(paramstruct,'Niter_inner') ;    
    Niter_inner = getfield(paramstruct,'Niter_inner') ; 
  end ;
  if isfield(paramstruct,'U_ini') ;    
    U = getfield(paramstruct,'U_ini') ; 
  end ;
  if isfield(paramstruct,'V_ini') ;    
    V = getfield(paramstruct,'V_ini') ; 
  end ;
  if isfield(paramstruct,'Omegav') ;    
    Omegav = getfield(paramstruct,'Omegav') ; 
    [VOmegav,DOmegav,~]=svds(Omegav,p-2);
    DOmegav=diag(DOmegav);
  end ;
  if isfield(paramstruct,'CVmethod') ; 
      CVmethod=getfield(paramstruct,'CVmethod') ; 
  end ;
end;






% define critical functions
if strcmpi(distr,'bernoulli') 
    distr='binomial'; % for using glmfit and glmnet
    fcn_b=@(theta)log(1+exp(theta));
    fcn_g=@(mu)log(mu./(1-mu));
    fcn_ginv=@(eta)exp(eta)./(1+exp(eta));
    fcn_db=@(theta)exp(theta)./(1+exp(theta));
    fcn_ddb=@(theta)exp(theta)./((1+exp(theta)).^2);
    fcn_dg=@(mu)1./(mu.*(1-mu));
elseif strcmpi(distr,'poisson')
    fcn_b=@(theta)exp(theta);
    fcn_g=@(mu)log(mu);
    fcn_ginv=@(eta)exp(eta);
    fcn_db=@(theta)exp(theta);
    fcn_ddb=@(theta)exp(theta);
    fcn_dg=@(mu)1./mu;
elseif strcmpi(distr,'normal')
    fcn_b=@(theta)(theta.^2)/2;
    fcn_g=@(mu)(mu);
    fcn_ginv=@(eta)(eta);
    fcn_db=@(theta)(theta);
    fcn_ddb=@(theta)ones(size(theta));
    fcn_dg=@(mu)ones(size(mu));
end;    






% initialize iteration
opt_lambda=zeros(niter_lam,r);
diff=inf;
niter=1;
while diff>Tol && niter<niter_lam   % use CV choose lambda, rec best lambda for each niter for each component (median over IRLS)
    
    V_old=V;
    
    for k=1:r % cycle through components
        uk=U(:,k);
        vk=V(:,k);
        fixdata=U*V'-uk*vk'; % treat as offset 
        tfixdata=fixdata';
        
        % fix vk, est uk 
        for i=1:n
%             uk(i) = glmfit(vk,tdata(:,i),distr,'constant','off','offset',tfixdata(:,i));        
            fit_u=glmnet(vk,tdata(:,i),distr,struct('offset',tfixdata(:,i),'alpha',0,'lambda',0.01,'intr',false));
            uk(i)=fit_u.beta;
        end;

 
        % fix uk, est vk 
        offset=fixdata(:);
        X=kron(eye(p),uk);
        Y=data(:);   
        niter_vk=1; 
        diff_vk=inf;
        rec_lam_temp=[];
        while diff_vk>Tol_inner && niter_vk<Niter_inner % IRLS, fix the max iter number for IRLS to be 10
            vk_old=vk;
            % calc weight matrix
            eta=X*vk+offset; % np*1
            mu=fcn_db(eta); %np*1
            W=1./(fcn_ddb(eta).*((fcn_dg(mu)).^2)); % diagonal of weight matrix for IRLS
            sw=sqrt(W); % sqrt of diagonal of weight matrix, np*1
            Xmat=bsxfun(@times,reshape(sw,n,p),uk); % n,p
            Ycurr=(eta-offset)+(Y-mu).*fcn_dg(mu); %np*1
            Ymat=reshape(sw.*Ycurr,n,p); %n*p
            if CVmethod==1
                [lambda,vk]=LeaveEntOut_GCV(Ymat,Xmat, VOmegav,DOmegav);
            elseif CVmethod==2
                [lambda,vk]=LeaveColOut_CV(Ymat,Xmat, VOmegav,DOmegav);
            end;
            rec_lam_temp=[rec_lam_temp,lambda];
            % update stopping rule
            niter_vk=niter_vk+1;
            diff_vk=PrinAngle(vk,vk_old);
        end;
       
        opt_lambda(niter,k)=median(rec_lam_temp);
    
%         % check (we hope the lines stablize, at least when niter is large they stablize)
%         % note: FPCA does not have this issue since fixing u est v would be
%         % a single penalized least square there instead of penalized IRLS
%         figure(10);
%         subplot(1,r,k)
%         plot(rec_lam_temp,'o-');
%         hold on;
%         xlabel('IRLS iterations')
%         ylabel(['Selected lambda_',num2str(k)]);
%         title(['For v_',num2str(k),', Niter=',num2str(niter)]);
%         drawnow;
%         figure(k);clf;
%         plot(vk);title(num2str(k));
%         k
        
        % normalize
        temp=norm(vk);
        U(:,k)=uk*temp;
        V(:,k)=vk/temp; % norm 1              
    end;
    
    % normalize
    [U,D,V]=svds(U*V',r);
    U=U*D;
    
    % calculate stopping rule
    diff=PrinAngle(V,V_old); % max principal angle (0~90)
    niter=niter+1;
end; 
    
    
% determine the best lambda for each component
lambdav=median(opt_lambda(1:(niter-1),:),1);
while diff>Tol && niter<Niter   % fixed tuning
    V_old=V;
    
    for k=1:r % cycle through components
        uk=U(:,k);
        vk=V(:,k);
        fixdata=U*V'-uk*vk'; % treat as offset 
        tfixdata=fixdata';
        
        % fix vk, est uk 
        for i=1:n
%             uk(i) = glmfit(vk,tdata(:,i),distr,'constant','off','offset',tfixdata(:,i));        
            fit_u=glmnet(vk,tdata(:,i),distr,struct('offset',tfixdata(:,i),'alpha',0,'lambda',0.01,'intr',false));
            uk(i)=fit_u.beta;      
        end;
        
        % fix uk, est vk 
        offset=fixdata(:);
        X=kron(eye(p),uk);
        Y=data(:); 
        niter_vk=1; 
        diff_vk=inf;
        while diff_vk>Tol_inner && niter_vk<Niter_inner % IRLS
            vk_old=vk;         
            % calc weight matrix
            eta=X*vk+offset; % np*1
            mu=fcn_db(eta); %np*1
            W=1./(fcn_ddb(eta).*((fcn_dg(mu)).^2)); % diagonal of weight matrix for IRLS
            sw=sqrt(W); % sqrt of diagonal of weight matrix, np*1
            XWX=sum((bsxfun(@times,reshape(sw,n,p),uk)).^2,1)';% diagonal values of the diagonal matrix, P*P
            % working response
            Ycurr=(eta-offset)+(Y-mu).*fcn_dg(mu); %np*1
            XWY=sum(bsxfun(@times,reshape(W,n,p),uk).*reshape(Ycurr,n,p),1)'; % X'*W*Ycurr, p*1
            lambda=lambdav(k);
            vk=woodbury(XWX,VOmegav,n*p*lambda*DOmegav)*XWY;  % np samples, so n*p*lambda
            
            % update stopping rule
            niter_vk=niter_vk+1;
            diff_vk=PrinAngle(vk,vk_old);
        end;
        
         % normalize
        temp=norm(vk);
        U(:,k)=uk*temp;
        V(:,k)=vk/temp; % norm 1
    end;
        

    % normalize
    [U,D,V]=svds(U*V',r);
    U=U*D;
    
    % calculate stopping rule
    diff=PrinAngle(V,V_old); % max principal angle (0~90)
    niter=niter+1;
    
end;




if niter==Niter
    disp(['EFPCA_oneway does NOT converge after ',num2str(Niter),' iterations!']);      
else
    disp(['EFPCA_oneway converges after ',num2str(niter),' iterations.']);      
end;




