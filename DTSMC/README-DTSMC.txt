### Instructions for the Density Tempered Sequential Monte Carlo (DTSMC) code in Matlab  ###

NOTE: the DTSMC algorithm was originally developed with a different naming scheme: AISIL
For this reason, the readme file and the code in this directory refer to AISIL rather than DTSMC. 

* LBA_AISIL_v1.m : This is the primary file to run the
  algorithm. Start here. This file loads subsidiary scripts as
  required, including:
   -> LBA_realdata.mat : Data from Forstmann et al. (2008) in a format ready for estimation with AISIL.
   -> compute_logpdf_y.m : calculates log p(y|theta, alpha).
   -> compute_cov_temp.m : calculates temporary covariance matrix for sampling.
   -> LBA_CMC_AIS_v1.m : Conditional Monte Carlo algorithm for updating estimates of the random effects in AISIL.
* AISIL_Forstmann_v1.mat : Posterior samples for Forstmann et al. (2008) estimated by LBA_AISIL_v1.m.

When applying AISIL to different data sets, the following scripts may
require editing depending on the design of the new data set and the
model that is estimated.
* LBA_n1PDF_reparam_real.m : LBA race equation for two-choice task. 
* LBA_tpdf.m : density function for a single accumulator.
* LBA_tcdf.m : distribution function for a single accumulator.
* reshape_v.m : assign appropriate drift rates (correct, error) to appropriate trials.
* reshape_b.m :  assign appropriate response thresholds (accuracy, neutral, speed) to appropriate trials. 

Subsidiary Matlab functions. These don't require editing for application to different data sets.
* logmvnpdf.m : log density of multivariate normal distribution.
* topdm.m : transform to positive definite symmetric matrix.
* rs_systematic.m : multinomial systematic resampling algorithm.
