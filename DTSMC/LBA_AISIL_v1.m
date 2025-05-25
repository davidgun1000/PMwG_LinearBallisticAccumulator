% estimating the hierarchical LBA model using AISIL method for the Forstmann (2008) dataset
% The LBA specification can be found in the paper: New Estimation
% approaches for the linear Ballistic Accumulator Model
% The data is stored in the matlab file 'LBA_realdata.mat', it has three
% components: 
% data.cond: the conditions of the experiments, we have three conditions in
% the Forstmann data
% data.rt: the response time
% data.response: response = 1 for incorrect response and response = 2 for correct response.

load('LBA_realdata.mat'); %load dataset
%parpool(28);  %number of processors available to be used.
num_subjects=length(data.rt); %number of subjects in the experiments
for j=1:num_subjects
    num_trials(j,1)=length(data.rt{j,1}); %computing the number of trials per subject
end
num_particles=100; %number of particles in the conditional Monte Carlo algorithm for the Markov move component

T=10000; %maximum number of annealing steps
M=250; %number of annealing particles
K=10; %fixed number of Markov moves at each annealing temperature for each AISIL samples 
num_randeffect=7; %total number of random effects (subject-level parameters) in the LBA model. For Forstmann dataset, we have 7 random effects.
%initial values of the hyperparameters for lower level parameters
prior_mu_mean=zeros(num_randeffect,1); %prior for the parameter \mu_{\alpha}
prior_mu_sig2=eye(num_randeffect);% prior for the parameter \Sigma_{\alpha}
v_half=2;%the hyperparameters of the prior of \Sigma_{\alpha}
A_half=1;%the hyperparameters of the prior of \Sigma_{\alpha}

% generate the initial AISIL samples from the prior density,
% see algorithm 4 (1) of the paper
particles_theta_mu=mvnrnd(prior_mu_mean',prior_mu_sig2,M); %initial AISIL samples for \mu_{\alpha}

particles_a1_half=1./random('gam',1/2,1,M,1); %initial AISIL samples for a_{1}
particles_a2_half=1./random('gam',1/2,1,M,1); %initial AISIL samples for a_{2}
particles_a3_half=1./random('gam',1/2,1,M,1); %initial AISIL samples for a_{3}
particles_a4_half=1./random('gam',1/2,1,M,1); %initial AISIL samples for a_{4}
particles_a5_half=1./random('gam',1/2,1,M,1); %initial AISIL samples for a_{5}
particles_a6_half=1./random('gam',1/2,1,M,1); %initial AISIL samples for a_{6}
particles_a7_half=1./random('gam',1/2,1,M,1); %initial AISIL samples for a_{7}

for i=1:M
    B_half=2*v_half*diag([1/particles_a1_half(i,1);1/particles_a2_half(i,1);1/particles_a3_half(i,1);1/particles_a4_half(i,1);1/particles_a5_half(i,1);1/particles_a6_half(i,1);1/particles_a7_half(i,1)]);
    k_half=v_half+num_randeffect-1;
    particles_theta_sig2(:,:,i)=iwishrnd(B_half,k_half); %initial AISIL samples for \Sigma_{\alpha}
end

%obtain initial AISIL samples for the random effects from the prior
%distribution p(\alpha|\theta) for each subject.
for i=1:M
    particles_theta_latent=mvnrnd(particles_theta_mu(i,:),particles_theta_sig2(:,:,i),num_subjects);
    particles_theta_latent_b1(i,:)=particles_theta_latent(:,1)';
    particles_theta_latent_b2(i,:)=particles_theta_latent(:,2)';
    particles_theta_latent_b3(i,:)=particles_theta_latent(:,3)';
    particles_theta_latent_A(i,:)=particles_theta_latent(:,4)';
    particles_theta_latent_v1(i,:)=particles_theta_latent(:,5)';
    particles_theta_latent_v2(i,:)=particles_theta_latent(:,6)';
    particles_theta_latent_tau(i,:)=particles_theta_latent(:,7)';    
end

% initialise AISIL settings, see algorithm 4 (1) of the paper
psisq=((0:T)./T).^3; % the sequence of tempering a_p
W=ones(M,1)./M; % equal weighted AISIL samples initially
log_llh=0;%initialise the marginal likelihood 
ESSall=zeros(T,1); % preallocate memory for the ESSall
psisq_current=psisq(1); 
t=2;
% compute the p(y|\theta,\alpha) for the initial AISIL samples
parfor i=1:M
      llh_calc(i,1)=compute_logpdf_y(data_mat,particles_theta_latent_b1(i,:)',...
          particles_theta_latent_b2(i,:)',particles_theta_latent_b3(i,:)',particles_theta_latent_A(i,:)',...
          particles_theta_latent_v1(i,:)',particles_theta_latent_v2(i,:)',...
          particles_theta_latent_tau(i,:)',num_subjects,num_trials);    
end

while t<=T+1
    t
    % there are three main steps in the AISIL algorithm, reweighting steps
    % for each of the AISIL samples 
     
    % (1) the first step of AISIL algorithm: reweighting step 
    incw=(psisq(t)-psisq_current).*llh_calc;
    max_incw=max(incw);
    w=exp(incw-max_incw); %computing the weight of each AISIL samples
    W=w./sum(w); %normalised the weight of AISIL samples
    ESS=1/sum(W.^2); %compute the ESS of the AISIL samples
    
    %---------------
    % Find the next annealing temperature by searching across grid of
    % points specify by variable psisq to maintain effective sample size
    % near some constant, in this case 0.8*M, then re-normalise weights.
    % See algorithm 4 (2b, 2c) of the paper.
    
    while ESS>=0.8*M
      t=t+1;
      if (t>=T+1)
	    t=T+1;
        incw=(psisq(t)-psisq_current).*llh_calc;
        max_incw=max(incw);
        w=exp(incw-max_incw);
        W=w./sum(w);
        ESS=1/sum(W.^2);
        break
      else
        incw=(psisq(t)-psisq_current).*llh_calc;
        max_incw=max(incw);
        w=exp(incw-max_incw);
        W=w./sum(w);
        ESS=1/sum(W.^2); 
      end
    end
    %---------------
    psisq_current=psisq(t); %set the current annealing temperature
    ESSall(t-1)=ESS; 
    log_llh=log_llh+log(mean(w))+max_incw; %compute the log of the marginal likelihood contribution at each annealing temperature.
    
    %computing the cholesky factor of the \Sigma_{\alpha} 
    for s=1:M
              chol_sigma_temp=chol(particles_theta_sig2(:,:,s),'lower');
              chol_theta_sig2_1(s,:)=[log(chol_sigma_temp(1,1))];
              chol_theta_sig2_2(s,:)=[chol_sigma_temp(2,1),log(chol_sigma_temp(2,2))];
              chol_theta_sig2_3(s,:)=[chol_sigma_temp(3,1:2),log(chol_sigma_temp(3,3))];
              chol_theta_sig2_4(s,:)=[chol_sigma_temp(4,1:3),log(chol_sigma_temp(4,4))];
              chol_theta_sig2_5(s,:)=[chol_sigma_temp(5,1:4),log(chol_sigma_temp(5,5))];
              chol_theta_sig2_6(s,:)=[chol_sigma_temp(6,1:5),log(chol_sigma_temp(6,6))];
              chol_theta_sig2_7(s,:)=[chol_sigma_temp(7,1:6),log(chol_sigma_temp(7,7))];   
    end
    
    % Training the proposals for the random effect for the Conditional
    % Monte Carlo algorithm in the Markov move component at each annealing
    % temperature
    for j=1:num_subjects    
        % In the matrix called theta_particles below, you have to list (1)
        % a list of vector of random effects particles in the LBA model, in
        % the case of Forstmann, you have \alpha_{b_1}, \alpha_{b_2},
        % \alpha_{b_3}, \alpha_A, \alpha_{v_1}, \alpha_{v_2},
        % \alpha_{tau}, (2) followed by the parameters \mu_{\alpha}, and
        % cholesky factor (lower triangular matrix) of the covariance
        % matrix \Sigma_{\alpha}
        theta_particles=[particles_theta_latent_b1(:,j),particles_theta_latent_b2(:,j),particles_theta_latent_b3(:,j),...
            particles_theta_latent_A(:,j),particles_theta_latent_v1(:,j),particles_theta_latent_v2(:,j),...
            particles_theta_latent_tau(:,j),particles_theta_mu,chol_theta_sig2_1,chol_theta_sig2_2,...
            chol_theta_sig2_3,chol_theta_sig2_4,chol_theta_sig2_5,chol_theta_sig2_6,chol_theta_sig2_7];
        
        length_param=size(theta_particles,2);
        mean_theta(j,:)=sum(theta_particles.*(W*ones(1,length_param))); %computing the weighted sample mean for the joint random effects and parameters \mu_{\alpha} and \Sigma_{\alpha}
        aux=theta_particles-ones(M,1)*mean_theta(j,:);
        covmat_theta(:,:,j)=aux'*diag(W)*aux; %compute the weighted sample covariance matrix for the joint random effects and parameters \mu_{\alpha} and \Sigma_{\alpha}
      
    end
    % (2) the second step for the AISIL method: resampling. 
    % See algorithm 4 (2d) of the paper.
    indx=rs_systematic(W'); % use systematic or multinomial resampling
    indx=indx';
    particles_theta_mu=particles_theta_mu(indx,:); % resample the AISIL samples for particles \mu_{\alpha} according to index 'indx'
    particles_theta_sig2=particles_theta_sig2(:,:,indx); % resample the AISIL samples for particles \Sigma_{\alpha} according to index 'indx'
    particles_a1_half=particles_a1_half(indx,:); % resample the AISIL samples for particles a_{1},..., a_{7} according to index 'indx'
    particles_a2_half=particles_a2_half(indx,:); % resample the AISIL samples for particles a_{1},..., a_{7} according to index 'indx'
    particles_a3_half=particles_a3_half(indx,:); % resample the AISIL samples for particles a_{1},..., a_{7} according to index 'indx'
    particles_a4_half=particles_a4_half(indx,:); % resample the AISIL samples for particles a_{1},..., a_{7} according to index 'indx'
    particles_a5_half=particles_a5_half(indx,:); % resample the AISIL samples for particles a_{1},..., a_{7} according to index 'indx'
    particles_a6_half=particles_a6_half(indx,:); % resample the AISIL samples for particles a_{1},..., a_{7} according to index 'indx'
    particles_a7_half=particles_a7_half(indx,:); % resample the AISIL samples for particles a_{1},..., a_{7} according to index 'indx'
    particles_theta_latent_b1=particles_theta_latent_b1(indx,:); %resample the AISIL samples for \alpha_(b_{1}) according to index 'indx'
    particles_theta_latent_b2=particles_theta_latent_b2(indx,:); %resample the AISIL samples for \alpha_(b_{2}) according to index 'indx'
    particles_theta_latent_b3=particles_theta_latent_b3(indx,:); %resample the AISIL samples for \alpha_(b_{3}) according to index 'indx'
    particles_theta_latent_A=particles_theta_latent_A(indx,:); %resample the AISIL samples for \alpha_(A) according to index 'indx'
    particles_theta_latent_v1=particles_theta_latent_v1(indx,:); %resample the AISIL samples for \alpha_(v_{1}) according to index 'indx'
    particles_theta_latent_v2=particles_theta_latent_v2(indx,:); %resample the AISIL samples for \alpha_(v_{2}) according to index 'indx'
    particles_theta_latent_tau=particles_theta_latent_tau(indx,:); %resample the AISIL samples for \alpha_(\tau) according to index 'indx'
    llh_calc=llh_calc(indx,:);
    W=ones(M,1)./M; % equal weight for each of the AISIL samples after resampling step.
    
    %The third step of AISIL algorithm: Markov moves step. This can be
    %done in parallel for each of the AISIL samples. For each AISIL samples, 
    % we have fixed number Markov moves K=10.
    % See algorithm 4 (2e) of the paper.
     
     parfor i=1:M  
        iter=1;
        while iter<=K
         
        %sample particles_theta_mu \mu_{\alpha} for each of the AISIL samples.
        var_mu=inv(num_subjects*inv(particles_theta_sig2(:,:,i))+inv(prior_mu_sig2));
        mean_mu=var_mu*(inv(particles_theta_sig2(:,:,i))*[sum(particles_theta_latent_b1(i,:));sum(particles_theta_latent_b2(i,:));sum(particles_theta_latent_b3(i,:));sum(particles_theta_latent_A(i,:));sum(particles_theta_latent_v1(i,:));...
            sum(particles_theta_latent_v2(i,:));sum(particles_theta_latent_tau(i,:))]);
        chol_var_mu=chol(var_mu,'lower');
        particles_theta_mu(i,:)=mvnrnd(mean_mu,chol_var_mu*chol_var_mu');
        
        %sample particles_theta_sig2 \Sigma_{\alpha} for each of the AISIL
        %samples
        k_half=v_half+num_randeffect-1+num_subjects;        
        [cov_temp]=compute_cov_temp(particles_theta_latent_b1(i,:),particles_theta_latent_b2(i,:),...
         particles_theta_latent_b3(i,:),particles_theta_latent_A(i,:),particles_theta_latent_v1(i,:),particles_theta_latent_v2(i,:),particles_theta_latent_tau(i,:),...
         particles_theta_mu(i,:),num_subjects);
        
        B_half=2*v_half*diag([1/particles_a1_half(i,1);1/particles_a2_half(i,1);1/particles_a3_half(i,1);1/particles_a4_half(i,1);...
            1/particles_a5_half(i,1);1/particles_a6_half(i,1);1/particles_a7_half(i,1)])+cov_temp;
        particles_theta_sig2(:,:,i)=iwishrnd(B_half,k_half);
        
        theta_sig2_inv=inv(particles_theta_sig2(:,:,i)); %computing the inverse of \Sigma_{\alpha}

        % The following computed separately for each of the 7 random
        % effects
        v1_half=(v_half+num_randeffect)/2;
        s1_half=(v_half*theta_sig2_inv(1,1)+A_half);
        particles_a1_half(i,1)=1./random('gam',v1_half,1/s1_half);
        
        v2_half=(v_half+num_randeffect)/2;
        s2_half=(v_half*theta_sig2_inv(2,2)+A_half);
        particles_a2_half(i,1)=1./random('gam',v2_half,1/s2_half);
    
        v3_half=(v_half+num_randeffect)/2;
        s3_half=(v_half*theta_sig2_inv(3,3)+A_half);
        particles_a3_half(i,1)=1./random('gam',v3_half,1/s3_half);

        v4_half=(v_half+num_randeffect)/2;
        s4_half=(v_half*theta_sig2_inv(4,4)+A_half);
        particles_a4_half(i,1)=1./random('gam',v4_half,1/s4_half);

        v5_half=(v_half+num_randeffect)/2;
        s5_half=(v_half*theta_sig2_inv(5,5)+A_half);
        particles_a5_half(i,1)=1./random('gam',v5_half,1/s5_half);

        v6_half=(v_half+num_randeffect)/2;
        s6_half=(v_half*theta_sig2_inv(6,6)+A_half);
        particles_a6_half(i,1)=1./random('gam',v6_half,1/s6_half);

        v7_half=(v_half+num_randeffect)/2;
        s7_half=(v_half*theta_sig2_inv(7,7)+A_half);
        particles_a7_half(i,1)=1./random('gam',v7_half,1/s7_half);
        
        %Conditional Monte Carlo step to update the random effects
        [particles_theta_latent_b1(i,:),particles_theta_latent_b2(i,:),...
             particles_theta_latent_b3(i,:),particles_theta_latent_A(i,:),...
             particles_theta_latent_v1(i,:),particles_theta_latent_v2(i,:),particles_theta_latent_tau(i,:),llh]=LBA_CMC_AIS_v1(data,...
             particles_theta_mu(i,:),particles_theta_sig2(:,:,i),...
             particles_theta_latent_b1(i,:)',particles_theta_latent_b2(i,:)',particles_theta_latent_b3(i,:)',...
             particles_theta_latent_A(i,:)',particles_theta_latent_v1(i,:)',particles_theta_latent_v2(i,:)',particles_theta_latent_tau(i,:)',...
             num_subjects,num_trials,num_particles,psisq_current,mean_theta,covmat_theta,num_randeffect);
         
        iter=iter+1;
        end
     end
%    % compute the p(y|\theta,\alpha) for the current AISIL samples 
     parfor i=1:M
          llh_calc(i,1)=compute_logpdf_y(data_mat,particles_theta_latent_b1(i,:)',...
              particles_theta_latent_b2(i,:)',particles_theta_latent_b3(i,:)',particles_theta_latent_A(i,:)',...
              particles_theta_latent_v1(i,:)',particles_theta_latent_v2(i,:)',...
              particles_theta_latent_tau(i,:)',num_subjects,num_trials);    
     end

     t=t+1;
     % save the output to your directory
     save('AISIL_Forstmann.mat','particles_theta_mu','particles_theta_sig2',...
          'particles_theta_latent_b1','particles_theta_latent_b2','particles_theta_latent_b3',...
          'particles_theta_latent_A','particles_theta_latent_v1','particles_theta_latent_v2','particles_theta_latent_tau',...
          'particles_a1_half','particles_a2_half','particles_a3_half','particles_a4_half','particles_a5_half','particles_a6_half','particles_a7_half',...
          'log_llh','W','t','ESSall');

end
