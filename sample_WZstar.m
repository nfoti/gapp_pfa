function [W Zstar] = sample_WZstar(params)
%  SAMPLE_WZstar Sample weights and auxiliary variables for probit thinning 
%    probabilities Znk.
%
%  Inputs: Extracted from params struct
%  Znk : NxK matrix of thinning indicators
%  W : (L+1)xK matrix of weights
%  Zstar : NxK matrix of auxiliary variables for probit model
%  Km : K element cell array, each entry is Nx(L+1) matrix (see
%       computeKernMats_exp2 function)
%  Lambda : (L+1)xK matrix with precisions of weights
%  
%  Returns : param struct with following fields updated
%     W : (L+1)xK matrix of weights
%     Zstar : NxK matrix with probit auxiliary variables

Znk = params.Znk;
W = params.W;
Zstar = params.Zstar;
Km = params.Km;
Lambda = params.Lambda;

jitter = 0.01; % From Radford Neal's MC implementation GP paper (thanks Sinead)
[N K] = size(Znk);

L = size(W,1)-1;

for k = 1:K
%for k = randperm(K)
  
  % Sample W
  K_k = Km{k};
  lambda = diag(Lambda(:,k));
  
  % We need Sigma anyways below, so why invert twice
  %Tau = lambda + K_k'*K_k + jitter*eye(L+1);
  %mu = Tau \ K_k'*Zstar(:,k);
  Sigma = inv(lambda + K_k'*K_k + jitter*eye(L+1));
  mu = Sigma*K_k'*Zstar(:,k);
  
  % Using L+1 for clarity, not efficiency
  %W(:,k) = mu + chol(Sigma)'*randn(L+1,1);
  % Use this because it is smarter than above
  W(:,k) = mvnrnd(mu', Sigma)';

  % !! HACK !!
  %W(1,k) = -mean(W(2:end,k));
  
  % Sample Zstar
  % truncate accordingly
  a = -inf(N,1);
  b = zeros(N,1);
  idx = Znk(:,k) == 1;
  a(idx) = 0;
  b(idx) = inf;
  
  Zstar(:,k) = randTN(K_k*W(:,k), 1, a, b);

end