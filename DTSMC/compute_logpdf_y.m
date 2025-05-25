function [llh_calc]=compute_logpdf_y(data_mat,particles_theta_latent_b1,...
    particles_theta_latent_b2,particles_theta_latent_b3,particles_theta_latent_A,particles_theta_latent_v1,particles_theta_latent_v2,...
    particles_theta_latent_tau,num_subjects,num_trials)
    
    %this function computes the log of p(y|\theta,\alpha)

    particles_theta_latent_b1_kron=[];
    particles_theta_latent_b2_kron=[];
    particles_theta_latent_b3_kron=[];    
    particles_theta_latent_A_kron=[];
    particles_theta_latent_v1_kron=[];
    particles_theta_latent_v2_kron=[];
    particles_theta_latent_tau_kron=[];

    for i=1:num_subjects
        particles_theta_latent_b1_kron=[particles_theta_latent_b1_kron;kron(particles_theta_latent_b1(i,1),ones(num_trials(i,1),1))];
        particles_theta_latent_b2_kron=[particles_theta_latent_b2_kron;kron(particles_theta_latent_b2(i,1),ones(num_trials(i,1),1))];
        particles_theta_latent_b3_kron=[particles_theta_latent_b3_kron;kron(particles_theta_latent_b3(i,1),ones(num_trials(i,1),1))];    
        particles_theta_latent_A_kron=[particles_theta_latent_A_kron;kron(particles_theta_latent_A(i,1),ones(num_trials(i,1),1))];
        particles_theta_latent_v1_kron=[particles_theta_latent_v1_kron;kron(particles_theta_latent_v1(i,1),ones(num_trials(i,1),1))];
        particles_theta_latent_v2_kron=[particles_theta_latent_v2_kron;kron(particles_theta_latent_v2(i,1),ones(num_trials(i,1),1))];
        particles_theta_latent_tau_kron=[particles_theta_latent_tau_kron;kron(particles_theta_latent_tau(i,1),ones(num_trials(i,1),1))];
    end
    
    % Note: 15818 (line 30) corresponds to the total length of the data 
    % variable across participants. 
    [particles_theta_latent_v_kron]=reshape_v(data_mat.response,particles_theta_latent_v1_kron,particles_theta_latent_v2_kron);   
    [particles_theta_latent_b_kron]=reshape_b(data_mat.cond,particles_theta_latent_b1_kron,particles_theta_latent_b2_kron,particles_theta_latent_b3_kron);
    llh_calc=sum(real(log(LBA_n1PDF_reparam_real(data_mat.rt, particles_theta_latent_A_kron, particles_theta_latent_b_kron, particles_theta_latent_v_kron, ...
        ones(15818,1),particles_theta_latent_tau_kron))));

end