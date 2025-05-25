function [cov_temp]=compute_cov_temp(particles_theta_latent_b1,particles_theta_latent_b2,...
    particles_theta_latent_b3,particles_theta_latent_A,particles_theta_latent_v1,particles_theta_latent_v2,particles_theta_latent_tau,...
    particles_theta_mu,num_subjects)

cov_temp=zeros(7,7);
for j=1:num_subjects
    theta_j=[particles_theta_latent_b1(1,j);particles_theta_latent_b2(1,j);...
             particles_theta_latent_b3(1,j);particles_theta_latent_A(1,j);particles_theta_latent_v1(1,j);...
             particles_theta_latent_v2(1,j);particles_theta_latent_tau(1,j)];  
    cov_temp=cov_temp+(theta_j-particles_theta_mu')*(theta_j-particles_theta_mu')';
         
end


end