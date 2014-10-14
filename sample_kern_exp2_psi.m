function [psi_inds] = sample_kern_exp2_psi(X,params)
%  SAMPLE_EXP_KERN_EXP2_psi Sample kernel parameters for each atom using a squared
%    exponential kernel.
%
%  Input: 
%  X : Nxd matrix of data covariates in rows
%  Following fields are stored in params struct
%  Znk : NxK matrix of thinning auxiliary variables
%  W : (L+1)xK matrix with weight vectors in columns
%  mus : Lxd matrix with fixed locations in rows
%  psi_inds : K vector with indices of atom dispersions
%  psi_dict : Dictionary with possible dispersion values in rows
%  U : vector same length as dictionaries containing the prior distribution
%      over locations and dispersions (default is uniform)
%
%  Notes:
%   - This function is specific to the squared exponential kernel!!
%   - It is possible that each atom could have a different psi for each
%     location but for now we'll assume a constant dispersion for each feature
%     over all locations. 

% Unpack data from structure
Znk = params.Znk;
W = params.W;
mus = params.mus;
psi_inds = params.psi_inds;
psi_dict = params.psi_dict;
U = params.U;

P = size(psi_dict,1);
[N K] = size(Znk);

if numel(U) == 1
  U = repmat(U,1,P);
end

for k = 1:K

  % Sample psi's
  znk = Znk(:,k);
  lp = zeros(1,P);

  % Not iterating over documents, this index is from the old code
  for p = 1:P

    % Update kernels using new dispersion
    K_k = [ones(N,1) exp(-pdist2(X,mus).^2/psi_dict(p))];
    Phi_k = normcdf(K_k*W(:,k));

    lp(p) = sum(znk.*log(Phi_k+eps) + (1-znk).*log(1-Phi_k+eps)); % likelihood
    lp(p) = lp(p) + log(U(p)); % prior

  end

  lp = lp - max(lp);
  pr = exp(lp);
  pr = pr ./ sum(pr);
  psi_inds(k) = discreternd(1,pr);

end
