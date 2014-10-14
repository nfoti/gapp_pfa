function [Xpn] = sample_Xpn(X, params)
% SAMPLE_XPN Sample component counts x_{pnk}.
%
% Inputs:
% X : "sparse" PxN matrix of word counts.  Stored as struct with following fields:
%     inds : Nx2 matrix with form [word_idx doc_idx]
%     vals : Nx1 vector of counts at each non-zero entry
%     P,N : Number of words and documents
% These inputs are extracted from params struct
% Znk : NxK matrix of thinning indicators
% Phi : PxK matrix of "topics", Phi(:,k) is probability vector
% Theta : 1xK vector of factor loadings (this can be multiplied by Znk to
%         thin these)
% Psi : KxN matrix of topic weights for each document
%
% Returns: 
% Xpn : struct representing sparse PxNxK tensor
%
%
% Note: Not using the sampled Znk right now but rather p(Znk = 1), i.e.
%       we're using the kernel GaP version here.
%
%       X.sum is the total number of words in the corpus.  This is the
%       maximum number of rows that Xpn.inds and Xpn.vals could have.

Znk = params.Znk;
Phi = params.Phi;
Theta = params.Theta;
Psi = params.Psi;
Km = params.Km;
W = params.W;

P = X.P;
N = X.N;
K = size(Phi,2);
NN = size(X.inds,1);

%inds = zeros(X.sum,3);
%vals = zeros(X.sum,1);

%run_sum = 0;

lTheta = log(Theta+eps);

lPhi = log(Phi + eps);
lPsi = log(Psi + eps);

% This version used the thinning indicators Znk
%lZnk = log(Znk + eps);
[inds,vals,run_sum] = sample_Xpn_meat(X.inds, X.vals, lPhi, ...
                                      lTheta, Znk, lPsi, X.sum);

% This version just puts the topic activation function in the likelihood
%[inds vals run_sum] = sample_Xpn_kernel_meat(X.inds, X.vals, lPhi, ...
%                                             lTheta, lPsi, Km, W, X.sum);


%{
%for ii = 1:NN
%  
%  p = X.inds(ii,1);
%  n = X.inds(ii,2);
%  xx = X.vals(ii);
%  
%  zeta = lPhi(p,:) + l_theta + lZnk(n,:);
%  zeta = zeta - max(zeta);
%  zeta = exp(zeta);
%  pp = bsxfun(@rdivide,zeta,sum(zeta));  % Don't call this p
%  if xx > 1
%    %mnr = mnrnd(xx, pp);
%    % Grabbed from mnrnd for speed
%    edges = [0 cumsum(pp)];
%    edges = min(edges,1); % guard histc against accumulated round-off, but after above check
%    edges(:,end) = 1; % get the upper edge exact
%    mnr = histc(rand(1,xx),edges,2);
%    
%    kk = find(mnr);
%    nk = numel(kk);
%  else
%    cump = cumsum(pp);
%    cump(end) = 1+eps;
%    kk = find((rand*(1-eps)) < cump, 1);
%    mnr(kk) = 1; % STUPID, but makes it work, writing C version anyways
%    nk = 1;
%  end
%  %inds = cat(1,inds,[p*ones(nk,1) n*ones(nk,1) kk']);
%  %vals = cat(1,vals,mnr(kk)');
%  %inds = [inds ; [p*ones(nk,1) n*ones(nk,1) kk']];
%  %vals = [vals ; mnr(kk)'];
%
%  idx1 = run_sum + 1;
%  idx2 = run_sum + nk;
%  %inds(idx1:idx2,:) = [p*ones(nk,1) n*ones(nk,1) kk'];
%  inds(idx1:idx2,1) = p;
%  inds(idx1:idx2,2) = n;
%  inds(idx1:idx2,3) = kk';
%  vals(idx1:idx2) = mnr(kk)';
%  run_sum = run_sum + nk;
% 
%end
%}

%% Pick back up
if run_sum < X.sum
  inds = inds(1:run_sum,:);
  vals = vals(1:run_sum);
end

% Get inds corresponding to each topic
kinds = cell(K,1);
kinds(1:max(inds(:,3))) = accumarray(uint32(inds(:,3)), 1:size(inds,1), [], @(x){x});
%tmp = accumarray(uint32(inds(:,3)), 1:size(inds,1), [], @(x){x});
%nmiss = K - numel(tmp);
%for ii = 1:nmiss
%  tmp{end+1} = [];
%end

% Copy into struct to return
Xpn = struct('inds',inds,'vals',vals,'kinds',{kinds},'size',[P N K]);
