function [Znk] = sample_Znk(params)
% SAMPLE_ZNK Gibbs update for thinning variables Znk
%
%  [Znk] = sample_Znk(T, params)
%
%  Inputs: Some taken from params struct
%   T : NxL matrix with covariates of data points in rows
%   Xpn : PxNxK sparse tensor of pseudo-counts for each topic
%   Theta : 1xK vector of factor loadings
%   Phi : PxK matrix of "topics", Phi(:,k) is probability vector
%   W : (L+1)xK matrix with vectors w_k in columns (for RVM)
%   Km : K-element cell array with kernel function for each feature

Xpn = params.Xpn;
Znk = params.Znk;
Theta = params.Theta;
Phi = params.Phi;
Psi = params.Psi;
Km = params.Km;
W = params.W;

N = size(Psi,2);
[P,K] = size(Phi);

inds = Xpn.inds;
vals = Xpn.vals;
kinds = Xpn.kinds;

rp = randperm(N);

Znk = sample_Znk_meat(Znk, Theta, Phi, Psi, Km, W, inds, vals, kinds, rp);

% Check that there is at least one 1 in each row of Znk
zinds = find(sum(Znk, 2) == 0);
zidx = randsample(K, numel(zinds), true);
Znk(sub2ind(size(Znk), zinds, zidx)) = 1;

