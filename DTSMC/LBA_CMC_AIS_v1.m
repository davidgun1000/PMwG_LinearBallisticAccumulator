function [theta_latent_b1,theta_latent_b2,theta_latent_b3,theta_latent_A,theta_latent_v1,theta_latent_v2,theta_latent_tau,llh]=LBA_CMC_AIS_v1(data,particles_theta_mu,...
particles_theta_sig2,theta_latent_b1,theta_latent_b2,theta_latent_b3,theta_latent_A,theta_latent_v1,theta_latent_v2,theta_latent_tau,...
num_subjects,num_trials,num_particles,psisq,mean_theta,covmat_theta,num_randeffect)
% this is the Conditional Monte Carlo algorithm to sample the random
% effects for each subjects
parfor j=1:num_subjects
    
    % generate the particles for the random effects
    % if annealing temperature (psisq) < 0.1, then we use prior density
    % p(\alpha|\theta) as a proposal density, otherwise, we use more
    % efficient mixture proposal as outlined in the Gunawan et al (2018). 
    
    if psisq<=0.1
    %generating particles for random effects from prior p(\alpha|\theta).    
    chol_covmat=chol(particles_theta_sig2,'lower'); %obtain the cholesky factor of the covariance matrix \Sigma_{\alpha}
    rnorm=particles_theta_mu'+chol_covmat*randn(num_randeffect,num_particles);
    rnorm=rnorm';
    else
    %generating particles for random effects from the mixture distribution as outlined in the Gunawan et al (2018)  
    %---------------------------------------------------------------------------
    w_mix=0.9; %setting the weight of the mixture for the proposal of random effects.
    u=rand(num_particles,1);
    id1=(u<w_mix);
    id2=1-id1;
    n1=sum(id1);
    n2=num_particles-n1;
    chol_theta_sig2=chol(particles_theta_sig2,'lower');
    chol_theta_sig2_1=log(chol_theta_sig2(1,1));
    chol_theta_sig2_2=[chol_theta_sig2(2,1),log(chol_theta_sig2(2,2))];
    chol_theta_sig2_3=[chol_theta_sig2(3,1:2),log(chol_theta_sig2(3,3))];
    chol_theta_sig2_4=[chol_theta_sig2(4,1:3),log(chol_theta_sig2(4,4))];
    chol_theta_sig2_5=[chol_theta_sig2(5,1:4),log(chol_theta_sig2(5,5))];
    chol_theta_sig2_6=[chol_theta_sig2(6,1:5),log(chol_theta_sig2(6,6))];
    chol_theta_sig2_7=[chol_theta_sig2(7,1:6),log(chol_theta_sig2(7,7))];
    xx=[particles_theta_mu';chol_theta_sig2_1';chol_theta_sig2_2';chol_theta_sig2_3';...
        chol_theta_sig2_4';chol_theta_sig2_5';chol_theta_sig2_6';chol_theta_sig2_7']; % we need this to compute the mean of the proposal 
    cond_mean=mean_theta(j,1:num_randeffect)'+covmat_theta(1:num_randeffect,num_randeffect+1:end,j)*((covmat_theta(num_randeffect+1:end,num_randeffect+1:end,j))\(xx-mean_theta(j,num_randeffect+1:end)')); % computing the mean of the proposal of random effects
    cond_var=covmat_theta(1:num_randeffect,1:num_randeffect,j)-covmat_theta(1:num_randeffect,num_randeffect+1:end,j)*(covmat_theta(num_randeffect+1:end,num_randeffect+1:end,j)\covmat_theta(num_randeffect+1:end,1:num_randeffect,j)); % computing the variance of the proposal of the random effects  
    chol_cond_var=chol(cond_var,'lower');
    rnorm1=cond_mean+chol_cond_var*randn(num_randeffect,n1);
    chol_covmat=chol(particles_theta_sig2,'lower');
    rnorm2=particles_theta_mu'+chol_covmat*randn(num_randeffect,n2);
    rnorm=[rnorm1,rnorm2];
    rnorm=rnorm';
    %------------------------------------------------------------------------------
    end
    
    rnorm_theta_b1=rnorm(:,1);
    rnorm_theta_b2=rnorm(:,2);
    rnorm_theta_b3=rnorm(:,3);
    rnorm_theta_A=rnorm(:,4);
    rnorm_theta_v1=rnorm(:,5);
    rnorm_theta_v2=rnorm(:,6);
    rnorm_theta_tau=rnorm(:,7);
    
    % set the first particles to the values of random effects from the
    % previous iterations for conditioning. For more details, look at the
    % paper.
    
    rnorm_theta_b1(1,1)=theta_latent_b1(j,1);
    rnorm_theta_b2(1,1)=theta_latent_b2(j,1);
    rnorm_theta_b3(1,1)=theta_latent_b3(j,1);
    rnorm_theta_A(1,1)=theta_latent_A(j,1);
    rnorm_theta_v1(1,1)=theta_latent_v1(j,1);
    rnorm_theta_v2(1,1)=theta_latent_v2(j,1);
    rnorm_theta_tau(1,1)=theta_latent_tau(j,1);

    rnorm_theta=[rnorm_theta_b1,rnorm_theta_b2,rnorm_theta_b3,rnorm_theta_A,rnorm_theta_v1,rnorm_theta_v2,rnorm_theta_tau];    
    
    %adjust the size of the vectors of the random effects
    
    rnorm_theta_b1_kron=kron(rnorm_theta_b1,ones(num_trials(j,1),1));
    rnorm_theta_b2_kron=kron(rnorm_theta_b2,ones(num_trials(j,1),1));
    rnorm_theta_b3_kron=kron(rnorm_theta_b3,ones(num_trials(j,1),1));
    rnorm_theta_A_kron=kron(rnorm_theta_A,ones(num_trials(j,1),1));
    rnorm_theta_v1_kron=kron(rnorm_theta_v1,ones(num_trials(j,1),1));
    rnorm_theta_v2_kron=kron(rnorm_theta_v2,ones(num_trials(j,1),1));
    rnorm_theta_tau_kron=kron(rnorm_theta_tau,ones(num_trials(j,1),1));
    
    %adjust the size of the dataset
    
    data_response_repmat=repmat(data.response{j,1}(:,1),num_particles,1);
    data_rt_repmat=repmat(data.rt{j,1}(:,1),num_particles,1);
    data_cond_repmat=repmat(data.cond{j,1}(:,1),num_particles,1);  
    
    [rnorm_theta_b_kron]=reshape_b(data_cond_repmat,rnorm_theta_b1_kron,rnorm_theta_b2_kron,rnorm_theta_b3_kron); % choose the threshold to match with the conditions of the experiments
    [rnorm_theta_v_kron]=reshape_v(data_response_repmat,rnorm_theta_v1_kron,rnorm_theta_v2_kron);% set the mean of the drift rate particles to match with the response data.
    
    %computing the log density of the LBA given the particles of the random
    %effects
    
    lw=real(log(LBA_n1PDF_reparam_real(data_rt_repmat, rnorm_theta_A_kron, rnorm_theta_b_kron, rnorm_theta_v_kron, ones(num_particles*num_trials(j,1),1),rnorm_theta_tau_kron)));
    lw_reshape=reshape(lw,num_trials(j,1),num_particles);
    logw_first=sum(lw_reshape);
    
    %computing the log of the weights
    %------------------------------
    if psisq<=0.1
    logw=psisq.*logw_first';
    else
        
    logw_second=(logmvnpdf(rnorm_theta,particles_theta_mu,particles_theta_sig2));
    logw_third=log(w_mix.*mvnpdf(rnorm_theta,cond_mean',chol_cond_var*chol_cond_var')+...
            (1-w_mix).*mvnpdf(rnorm_theta,particles_theta_mu,particles_theta_sig2));
    logw=psisq.*logw_first'+logw_second'-logw_third;
    end
    %------------------
    
    %check if there is imaginary number of logw
    id=imag(logw)~=0;
    id=1-id;
    id=logical(id);
    logw=logw(id,1); 
    logw=real(logw);

    if sum(isinf(logw))>0 | sum(isnan(logw))>0
     id=isinf(logw) | isnan(logw);
     id=1-id;
     id=logical(id);
     logw=logw(id,1);
    end
    
    max_logw=max(real(logw));
    weight=real(exp(logw-max_logw));
    llh_i(j) = max_logw+log(mean(weight)); 
    llh_i(j) = real(llh_i(j)); 	
    weight=weight./sum(weight);
    if sum(weight<0)>0
        id=weight<0;
        id=1-id;
        id=logical(id);
        weight=weight(id,1);
    end
    Nw=length(weight);
    
    if Nw>0 
        ind=randsample(Nw,1,true,weight);
        theta_latent_b1(j,1)=rnorm_theta(ind,1);
        theta_latent_b2(j,1)=rnorm_theta(ind,2);
        theta_latent_b3(j,1)=rnorm_theta(ind,3);
        theta_latent_A(j,1)=rnorm_theta(ind,4);
        theta_latent_v1(j,1)=rnorm_theta(ind,5);
        theta_latent_v2(j,1)=rnorm_theta(ind,6);
        theta_latent_tau(j,1)=rnorm_theta(ind,7);
    end
        
%----------------------------------------------------------------------------------------------------------------------------------    
    
end
llh=sum(llh_i);
theta_latent_b1=theta_latent_b1';
theta_latent_b2=theta_latent_b2';
theta_latent_b3=theta_latent_b3';
theta_latent_A=theta_latent_A';
theta_latent_v1=theta_latent_v1';
theta_latent_v2=theta_latent_v2';
theta_latent_tau=theta_latent_tau';
end

