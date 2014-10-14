function [Km] = computeKernMats_exp2(X,params)
%  COMPUTEKERNMATS Compute matrices of squared exponential kernel evaluated at 
%    data covariates and fixed locations for each feature.  Prepends a column 
%    of 1's to each matrix.
%
%  Inputs:
%  X : NxT matrix of covariate values for each data point
%  These inputs are in params struct
%  K : Number of topics
%  mus : Lxd matrix with atom covariate locations
%  psi_inds : K vector with indices of atom dispersions
%  psi_dict : Dictionary with possible kernel dispersions

K = size(params.Psi,1);
mus = params.mus;
psi_inds = params.psi_inds;
psi_dict = params.psi_dict;

N = size(X,1);
Km = cell(1,K);

for k = 1:K
  
  T = exp(-(pdist2(X,mus).^2)/psi_dict(psi_inds(k)));
  Km{k} = [ones(N,1) T];
  
end