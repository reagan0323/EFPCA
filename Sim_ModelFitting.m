% This file demonstrates the parameter estimation performance of the
% proposed methods. We consider Bernoulli, Binomial, and Poisson data.
%
% Settings:
% 1. Bernoulli data, one-way smooth
% 2. Poisson data, one-way smooth
% 3. Binomial data, two-way smooth
% 4. Bernoulli data, two-way smooth
% 5. Binomial data with unbalanced N, two-way smooth
%
% Contact: Gen Li, PhD
%          Assistant Professor of Biostatistics, Columbia University
%          Email: gl2521@columbia.edu  
%
% CopyRight all reserved
% Last updated: 2/3/2017

%% simulation setting

addpath efpca_matlab
addpath glmnet_matlab
Nsim=100;
n=100;
p=50;
r=2; 
xgrid=(1:p)/p; % 0~1, evenly spaced 
ygrid=(1:n)/n;


choosesetting=5;
switch choosesetting

    case 1 % one-way bernoulli 
        simname='Oneway_Bernoulli';
        distr='bernoulli';
        trueV=GramSchmidt([(xgrid.^2-xgrid+1/8)', sin(xgrid*2*pi)']);
        rng(100);
        temp=randn(n,2);
        temp=bsxfun(@minus,temp,mean(temp,1));
        trueU=GramSchmidt(temp)*diag([40,30]);
        trueGP=trueU*trueV';
        Theta=trueGP;
        param= exp(Theta)./(1+exp(Theta)); 
        figure();mesh(param) 
        N=zeros(n,p);

    case 2 % one-way poisson 
        simname='Oneway_Poisson';
        distr='poisson';
        trueV=GramSchmidt([(xgrid.^2-xgrid+1/8)', sin(xgrid*2*pi)']); 
        rng(100);
        temp=randn(n,2);
        temp=bsxfun(@minus,temp,mean(temp,1));
        trueU=GramSchmidt(temp)*diag([40,30]);
        trueGP=trueU*trueV';
        Theta=trueGP;
        param= exp(Theta); 
        figure();mesh(param)     
        N=zeros(n,p);
         
    case 3 % two-way binomial 
        simname='Twoway_Binomial';
        distr='binomial';
        trueV=GramSchmidt([(xgrid.^2-xgrid+1/8)', sin(xgrid*2*pi)']); 
        trueU=GramSchmidt([sin(ygrid*2*pi)',cos(ygrid*5*pi)']);
        trueD=[40,30];
        Theta=trueU*diag(trueD)*trueV';
        param= exp(Theta)./(1+exp(Theta)); 
        N=round(rand(n,p)*5+5); % random from 5 to 10. If N=1, it's exactly bernoulli
        figure();mesh(param)            
        
     case 4 % two-way bernoulli (generate data from our model)
         simname='Twoway_Bernoulli';
         distr='bernoulli';
         trueV=GramSchmidt([(xgrid.^2-xgrid+1/8)', sin(xgrid*2*pi)']); 
         trueU=GramSchmidt([sin(ygrid*2*pi)',cos(ygrid*5*pi)']);
         trueD=[40,30];
         Theta=trueU*diag(trueD)*trueV';
         param= exp(Theta)./(1+exp(Theta)); 
         N=ones(n,p);
         figure();mesh(param) 
         
     case 5 % a variant of two-way binomial 
        simname='Twoway_Binomial_inbalanceN';
        distr='binomial';
        trueV=GramSchmidt([(xgrid.^2-xgrid+1/8)', sin(xgrid*2*pi)']); 
        trueU=GramSchmidt([sin(ygrid*2*pi)',cos(ygrid*5*pi)']);
        trueD=[40,30];
        Theta=trueU*diag(trueD)*trueV';
        param= exp(Theta)./(1+exp(Theta)); 
        N=[round(rand(n/2,p/2)*10+10),round(rand(n/2,p/2)*2+2);...
            round(rand(n/2,p/2)*2+2),round(rand(n/2,p/2)*10+10)];
        figure();mesh(param)   
end;
simname 




%% run the proposed low-rank exponential family methods
if strcmpi(distr,'bernoulli')
    X=binornd(1,param);
elseif strcmpi(distr,'poisson')
    X=poissrnd(param);
elseif strcmpi(distr,'binomial')
    X=binornd(N,param);
end;
% 
if choosesetting==1 || choosesetting==2

    [U_regmaf1,V_regmaf1]=EFPCA_oneway3(X,distr, r); 
    Theta_regmaf1=U_regmaf1*V_regmaf1';
    V_regmaf1=CorrectSign(trueV,V_regmaf1);

    % compare V
    figure(1);clf;
    plot(trueV,'k-');
    hold on
    plot(V_regmaf1,'b--');

elseif choosesetting==3 || choosesetting==5
    [U_regmaf1,V_regmaf1]=EFPCA_binomial_twoway3(X,N,r); 
    Theta_regmaf1=U_regmaf1*V_regmaf1';
    [U_temp,D_temp,V_temp]=svds(Theta_regmaf1,r);
    
    % bootstrap for CI (here bootstrap sample set to 5 for illustration purpose only)
    [Uarray,Darray,Varray]=Param_Bootstrap_EFPCA(U_temp,diag(D_temp),V_temp,distr,2,struct('Nboot',5,'N',N));

    % compare V
    figure(1);clf;
    plot(trueV,'k-');
    hold on
    plot(V_temp,'r-');
    for i=1:5
        plot(Varray(:,:,i),'c--');
    end;
    % compare U
    figure(2);clf;
    plot(trueU,'k-');
    hold on
    plot(U_temp,'r-');
    for i=1:5
        plot(Uarray(:,:,i),'c--');
    end;

elseif choosesetting==4
    [U_regmaf1,V_regmaf1]=EFPCA_twoway3(X,distr,r,struct('CVmethod',2));  
    Theta_regmaf1=U_regmaf1*V_regmaf1';
    [U_temp,D_temp,V_temp]=svds(Theta_regmaf1,r);
    
    % bootstrap for CI (here bootstrap sample set to 5 for illustration purpose only)
    [Uarray,Darray,Varray]=Param_Bootstrap_EFPCA(U_temp,diag(D_temp),V_temp,distr,2,struct('Nboot',5,'N',N));

    % compare V
    figure(1);clf;
    plot(trueV,'k-');
    hold on
    plot(V_temp,'r-');
    for i=1:5
        plot(Varray(:,:,i),'c--');
    end;
    % compare U
    figure(2);clf;
    plot(trueU,'k-');
    hold on
    plot(U_temp,'r-');
    for i=1:5
        plot(Uarray(:,:,i),'c--');
    end;
end;

