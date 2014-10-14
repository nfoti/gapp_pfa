function [Phi] = sample_Phi(params)
%  SAMPLE_PHI Gibbs update for Dirichlet parameters \phi_k.
%
%  Inputs: extracted from params struct
%  Ypn : struct representing sparse tensor of factor counts (see sample_Xpn).
%        Members 'inds', and 'vals', 'kinds'
%        inds : Lx3 vector with word, document and factor inds
%        vals : Lx1 vector with the pseudo-counts
%        kinds : cell array with K entries, entry k has indices into 'inds' 
%                 and 'vals' where inds(:,3) == k (computed with
%                 accumarray(inds(:,3),1:size(inds,1),[],@(x){x}) 
%                 ) which should be done outside b/c will need it for 
%                 another function as well
%        size : dimensions of tensor (PxNxK)
%  alpha_phi : Scalar parameter of Dirichlet prior
%
%  Returns: PxK matrix Phi with updated topics
%

Xpn = params.Xpn;
alpha_phi = params.alpha_phi;

P = Xpn.size(1);
N = Xpn.size(2);
K = Xpn.size(3);

inds = Xpn.inds;
vals = Xpn.vals;
kinds = Xpn.kinds;

Phi = zeros(P,K);

for k = 1:K
   
  % Instantiate Xk
  % This is going to be VERY SLOW for now
  kk = kinds{k};
  if numel(kk) > 0
    Xk = accumarray(inds(kk,1:2), vals(kk), [P N]);
    sumX = sum(Xk,2);
  else
    sumX = zeros(P,1);
  end

  % Draw Dirichlet random variable
  tt = alpha_phi + sumX;
  y = randg(tt);
  Phi(:,k) = y ./ sum(y);
  
end
