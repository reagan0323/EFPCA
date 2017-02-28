function [U,V,lambdau,lambdav]=EFPCA_binomial_twoway3(data,N,r, paramstruct)
% This is two-way EFPCA function for truly binomial data.
% It decomposes multi-rank two-way smooth natural parameter matrix.
% Loadings and scores are both smooth (with different degrees of smoothness for different ranks)
% A sub-case for EFPCA_twoway3.m
%
% input: 
%
%   data    n*p data matrix, number of success trials
%
%   N       n*p data matrix, number of total trials
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
%           Omegau      n*n matrix for roughtness penalty, default is classic
%                       smoothing spline Omega matrix
%
%           CVmethod    scalar, 1(default) is leave-one-entry-out CV
%                       2 is leave-one-column-out CV (smoother estimates)
%
%           fig         scalar, 0(default) no figure output
%                       1 show tuning selection figures over iterations
%
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
%
% 8.14.2016 by Gen Li

[n,p]=size(data);
tdata=data';
[n_,p_]=size(N);
tN=N';
% check
if r>min(n,p)
    error('Rank too large!')
elseif n_~=n || p_~=p
    error('Mismatching input for binomial!');
elseif sum(sum(data>N))>0
    error('# success cannot exceed # trials!');
end;


% initial values
Tol=0.1; % overall threshold
Tol_inner=0.1; % threshold for each component 
Niter=200;
Niter_inner=10; % max number of iterations for each penalized IRLS problem
niter_lam=20; % max number of outer iterations with floating lambdas
CVmethod=1; % leave-one-entry out CV
fig=0; % whether to show tuning selection figures
% V=randn(p,r); % default start
% U=randn(n,r);
tempdata=min(max(data,0.5),N-0.5); % bound data away from 0 and N
probini=tempdata./N; % raw est of prob matrix, no 0 or 1
thetaini=log(probini./(1-probini)); % no degeneracy
[U,D,V]=svds(thetaini,r);
U=U*D; % unique way to set initial value for binomial data
[~,VOmegav,DOmegav]=GetOmega(p);
[~,VOmegau,DOmegau]=GetOmega(n);
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
  if isfield(paramstruct,'niter_lam') ;    
    niter_lam = getfield(paramstruct,'niter_lam') ; 
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
  if isfield(paramstruct,'Omegau') ;    
    Omegau = getfield(paramstruct,'Omegau') ; 
    [VOmegau,DOmegau,~]=svds(Omegau,n-2);
    DOmegau=diag(DOmegau);
  end ;
  if isfield(paramstruct,'CVmethod') ; 
      CVmethod=getfield(paramstruct,'CVmethod') ; 
  end ;
  if isfield(paramstruct,'fig') ; 
      fig=getfield(paramstruct,'fig') ; 
  end ;
end;





% define critical functions for binomial distribution
% "total" represents the total number of trials, i.e., N
fcn_b=@(theta,total)total.*log(1+exp(theta));
fcn_g=@(mu,total)log(mu./(total-mu));
fcn_db=@(theta,total)(total.*exp(theta))./(1+exp(theta));
fcn_ddb=@(theta,total)(total.*exp(theta))./((1+exp(theta)).^2);
fcn_dg=@(mu,total)total./(mu.*(total-mu));






% floating lambda iterations
if fig
    figure(11);clf;
    lam_colormap=colormap(jet);
    nline=niter_lam;
    temp=floor(length(lam_colormap)/nline);
    lam_colormap=lam_colormap((0:(nline-1))*temp+1,:);
end;
%
opt_lambdau=zeros(niter_lam,r);
opt_lambdav=zeros(niter_lam,r);
diff=inf;
niter=1;
while diff>Tol && niter<=niter_lam   % let GCV choose lambda, rec best lambda for each niter for each component (median over IRLS)
    V_old=V;   
    for k=1:r % cycle through components
        uk=U(:,k);
        vk=V(:,k);
        offsetmat=U*V'-uk*vk'; 
        toffsetmat=offsetmat';
        
        % Est uk 
        vk=vk/norm(vk); % reset vk's norm to be 1
        X=kron(eye(n),vk);
        Y=tdata(:);
        Total=tN(:); % np*1
        offset=toffsetmat(:);
        niter_uk=1; 
        diff_uk=inf;
        rec_lam_temp=[]; % record GCV-selected lambda for each IRLS iteration
        while diff_uk>Tol_inner && niter_uk<Niter_inner % IRLS+penalty+GCV
            uk_old=uk;           
            % calc weight matrix
            eta=X*uk+offset; % np*1
            mu=fcn_db(eta,Total); %np*1
            W=1./(fcn_ddb(eta,Total).*((fcn_dg(mu,Total)).^2)); % diagonal of weight matrix for IRLS
            sw=sqrt(W); % sqrt of diagonal of weight matrix, np*1
            Xmat=bsxfun(@times,reshape(sw,p,n),vk); % p*n
            Ycurr=(eta-offset)+(Y-mu).*fcn_dg(mu,Total); %np*1
            Ymat=reshape(sw.*Ycurr,p,n); %p*n
            if CVmethod==1
                [lambda,uk]=LeaveEntOut_GCV(Ymat,Xmat, VOmegau,DOmegau);
            elseif CVmethod==2
                [lambda,uk]=LeaveColOut_CV(Ymat,Xmat, VOmegau,DOmegau);
            end;
            rec_lam_temp=[rec_lam_temp,lambda];
            % update stopping rule
            niter_uk=niter_uk+1;
            diff_uk=PrinAngle(uk,uk_old);
        end;
        % record the best lambda for this component for this iteration
        opt_lambdau(niter,k)=median(rec_lam_temp);
        
        if fig
            figure(11);
            subplot(2,r,k)
            plot(rec_lam_temp,'o-','color',lam_colormap(niter,:));
            hold on;
            xlabel('IRLS iterations')
            ylabel('Selected lambda_u in 10E[-5,5]')
            title(['u_',num2str(k),', Niter=',num2str(niter)]);
        end;

        
        % Est vk 
        uk=uk/norm(uk); % normalize uk with norm 1
        X=kron(eye(p),uk);
        Y=data(:);   
        Total=N(:); % np*1
        offset=offsetmat(:);
        niter_vk=1; 
        diff_vk=inf;
        rec_lam_temp=[];
        while diff_vk>Tol_inner && niter_vk<Niter_inner % IRLS+penalty+GCV
            vk_old=vk;     
            % calc weight matrix
            eta=X*vk+offset; % np*1
            mu=fcn_db(eta,Total); %np*1
            W=1./(fcn_ddb(eta,Total).*((fcn_dg(mu,Total)).^2)); % diagonal of weight matrix for IRLS
            sw=sqrt(W); % sqrt of diagonal of weight matrix, np*1
            Xmat=bsxfun(@times,reshape(sw,n,p),uk); % n,p
            Ycurr=(eta-offset)+(Y-mu).*fcn_dg(mu,Total); %np*1
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
        opt_lambdav(niter,k)=median(rec_lam_temp);
    
        if fig
            figure(11);
            subplot(2,r,r+k)
            plot(rec_lam_temp,'o-','color',lam_colormap(niter,:));
            hold on;
            xlabel('IRLS iterations')
            ylabel('Selected lambda_v in 10E[-5,5]')
            title(['v_',num2str(k),', Niter=',num2str(niter)]);
            drawnow
        end;
        
        % store in U and V
        U(:,k)=uk; % unit norm
        V(:,k)=vk; % not unit norm
    end;
    
    % normalize
    [U,D,V]=svds(U*V',r);
    U=U*D;
    
    % calculate stopping rule
    diff=PrinAngle(V,V_old); % max principal angle (0~90)
    niter=niter+1;
end;
  





% determine the best lambda for each component
lambdau=median(opt_lambdau(1:(niter-1),:),1); 
lambdav=median(opt_lambdav(1:(niter-1),:),1);
if fig % plot the best tuning parameter for each rank for each loading
    figure(11);
    for k=1:r
        subplot(2,r,k)
        plot([1:3],ones(1,3)*lambdau(k),'k--','linewidth',1.5);
        subplot(2,r,r+k)
        plot([1:3],ones(1,3)*lambdav(k),'k--','linewidth',1.5);
    end;
end;
% fixed lambda iterations
while diff>Tol && niter<Niter   % fixed tuning
    V_old=V;     
    for k=1:r % cycle through components
        uk=U(:,k);
        vk=V(:,k);
        offsetmat=U*V'-uk*vk'; 
        toffsetmat=offsetmat';
        
        % Est uk 
        vk=vk/norm(vk);
        X=kron(eye(n),vk);
        Y=tdata(:);
        Total=tN(:);
        offset=toffsetmat(:);
        niter_uk=1; 
        diff_uk=inf;
        while diff_uk>Tol_inner && niter_uk<Niter_inner % IRLS+penalty
            uk_old=uk;        
            % fit one iteration of penalized weighted least square
            eta=X*uk+offset; % np*1
            mu=fcn_db(eta,Total); %np*1
            W=1./(fcn_ddb(eta,Total).*((fcn_dg(mu,Total)).^2)); % diagonal of weight matrix for IRLS
            sw=sqrt(W); % sqrt of diagonal of weight matrix, np*1
            XWX=sum((bsxfun(@times,reshape(sw,p,n),vk)).^2,1)';% diagonal values of the diagonal matrix, n*n
            Ycurr=(eta-offset)+(Y-mu).*fcn_dg(mu,Total); %np*1
            XWY=sum(bsxfun(@times,reshape(W,p,n),vk).*reshape(Ycurr,p,n),1)'; % X'*W*Ycurr, n*1
            uk=woodbury(XWX,VOmegau,(n*p)*lambdau(k)*DOmegau)*XWY;  
            % update stopping rule
            niter_uk=niter_uk+1;
            diff_uk=PrinAngle(uk,uk_old);
        end;

        
        % Est vk 
        uk=uk/norm(uk);
        X=kron(eye(p),uk);
        Y=data(:);   
        Total=N(:);
        offset=offsetmat(:);
        niter_vk=1; 
        diff_vk=inf;
        while diff_vk>Tol_inner && niter_vk<Niter_inner % IRLS+penalty
            vk_old=vk;     
            % fit one iteration of penalized weighted least square
            eta=X*vk+offset; % np*1
            mu=fcn_db(eta,Total); %np*1
            W=1./(fcn_ddb(eta,Total).*((fcn_dg(mu,Total)).^2)); % diagonal of weight matrix for IRLS
            sw=sqrt(W); % sqrt of diagonal of weight matrix, np*1
            XWX=sum((bsxfun(@times,reshape(sw,n,p),uk)).^2,1)';% diagonal values of the diagonal matrix, P*P
            Ycurr=(eta-offset)+(Y-mu).*fcn_dg(mu,Total); %np*1
            XWY=sum(bsxfun(@times,reshape(W,n,p),uk).*reshape(Ycurr,n,p),1)'; % X'*W*Ycurr, p*1
            vk=woodbury(XWX,VOmegav,n*p*lambdav(k)*DOmegav)*XWY;  % np samples, so n*p*lambda
            % update stopping rule
            niter_vk=niter_vk+1;
            diff_vk=PrinAngle(vk,vk_old);
        end;
        
        % store in U and V
        U(:,k)=uk; % unit norm
        V(:,k)=vk; % not unit norm
    end;
    
    % normalize
    [U,D,V]=svds(U*V',r);
    U=U*D;

    % calculate stopping rule
    diff=PrinAngle(V,V_old); % max principal angle (0~90)
    niter=niter+1;
end;



if niter==Niter
    disp(['EFPCA_twoway does NOT converge after ',num2str(Niter),' iterations!']);      
else
    disp(['EFPCA_twoway converges after ',num2str(niter),' iterations.']);      
end;





