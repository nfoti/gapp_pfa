function [params] = gapp_pfa_finite(X,T,Xtest,params)
%  GAPP_PFA_FINITE Sample thinned gamma-Poisson factor model using Gibbs 
%   sampling.
%
%   [] = gapp_pfa_finite(X)
%     X : "sparse" PxN matrix of training word counts.  Assume there are 
%         NN non-zero entries of the count matrix.  It is a struct 
%         with fields:
%           inds (NNx2) : matrix rows of form [word ind, doc ind]
%           vals (NNx1) : vector of word counts
%           P, N, sum : number of words, docs and total number of words
%                       respectively
%     T : Nxd matrix of covariate values for documents
%     Xtest : "sparse" PxN matrix of test word counts
%     params : struct that describes state of the sampler
%     

if nargin < 3
  error('Must initialize sampler with ''params'' struct');
end

% Compile mex functions in here in case it's not done
% compile_mex_functions;

P = X.P;
N = X.N;

% It will be convenient to have an actual sparse version of the data as
% well
if params.computeperplex
    %X_sp = sparse(X.inds(:,1), X.inds(:,2), X.vals, P, N);
    Xtest_sp = sparse(Xtest.inds(:,1), Xtest.inds(:,2), Xtest.vals);
    %Yflagtr = X_sp > 0;
    Yflagtest = Xtest_sp > 0;
end

addpath(genpath('~/matlab/lightspeed2012b'));

% Initialize and validate initial state (i.e. make fields for collected samples)
validate_sampler(params);

if params.rng_seed > -1
  s = RandStream(params.rng_type, 'Seed', params.rng_seed);
  RandStream.setGlobalStream(s);
end

[K,N] = size(params.Psi);
%P = size(params.Phi,1);

% Initialize kernel matrix
if params.dosampleZnk
    params.Km = computeKernMats_exp2(T, params);
else
    params.Km = {};
end
params.Xpn = sample_Xpn(X, params);
% NN = size(X.inds,1);
% inds = [X.inds ones(NN,1)];
% vals = X.vals;
% kinds = cell(K,1);
% kinds(1:max(inds(:,3))) = accumarray(uint32(inds(:,3)), 1:size(inds,1), [], @(x){x});
%params.Xpn = struct('inds',inds,'vals',vals,'kinds',{kinds},'size',[P N 1]);

if params.Znk == -1
  if params.dosampleZnk
    params.Znk = zeros(N, K);
    params.Znk = sample_Znk(params);
    %params.Znk = sample_Znk_slow(params);
  else
    params.Znk = ones(N, K);
  end
end
% Probit aux variables for sampling W
% Znk's that are 0 have negative Zstar and 1 have positive Zstar
params.Zstar = params.Znk - 0.5;

nburn = params.nburn;
nsamp = params.nsamp;
thin = params.thin;
verbose = params.verbose;
NSWEEP = nburn + nsamp*thin;
printfreq = params.printfreq;

params.lpiter = zeros(1, NSWEEP);
params.lpsamp = zeros(1, nsamp);
params.maxlp = -realmax;
params.PoissMeanSS = zeros(P,N);
params.countSS = 0;
params.perplex_vec = zeros(1,nsamp);

% These variables now hold the max prob. sample.
% initialize space for samples (only if sampling a series of variables)
% Theta_s = cell(1, nsamp);
% Phi_s = cell(1, nsamp);
% Psi_s = cell(1, nsamp);
% Znk_s = cell(1, nsamp);
% W_s = cell(1, nsamp);
% Lambda_s = cell(1, nsamp);
% psi_inds_s = cell(1, nsamp);

smp_name = params.smp_name;
logger = params.logger;

% Sample
sidx = 1;
saveid = 1;

if logger
  logfile = fopen([smp_name '.log'], 'w');
else
  logfile = 1;  % Set to stdout if verbose by not logger
end

% Used as temporary when computing kernels below
phiMat = zeros(N,K);

for sweep = 1:NSWEEP
  
  if (verbose || logger) && mod(sweep,printfreq) == 0
    fprintf(logfile, 'Sweep: %d of %d, ', sweep, NSWEEP);
  end
  
  % pseudocounts Xpn
  if params.dosampleXpn
    params.Xpn = sample_Xpn(X, params);
  end

  % thinning indicators Znk
  if params.dosampleZnk
    params.Znk = sample_Znk(params);
    %params.Znk = sample_Znk_slow(params);
  end

  % factor loadings Theta_k
  if params.dosampleTheta
    params.Theta = sample_Theta(params);
  end
  
  % topic weights for documents, Psi_nk
  if params.dosamplePsi
    params.Psi = sample_Psi(params);
  end
  
  % topics Phi_k
  if params.dosamplePhi
    params.Phi = sample_Phi(params);
  end

  % RVM kernel widths psi
  if params.sampleKernWidth
    params.psi_inds = sample_kern_exp2_psi(T, params);
    params = rmfield(params, 'Km');
    params.Km = computeKernMats_exp2(T, params);
  end
  
  % RVM parameters W, Z_star
  if params.dosampleW
    [params.W,params.Zstar] = sample_WZstar(params);
  end

  % RVM precisions
  if params.dosampleLambda
    params.Lambda = sample_lambda(params.W, params.c0, params.d0);
  end
  
  % log-probability of model
  params.lpiter(sweep) = logP(X, params);
  
  % print some more info
  if (verbose || logger) && mod(sweep,printfreq) == 0
    fprintf(logfile, 'K: %d, logp: %.2f\n', numel(unique(params.Xpn.inds(:,3))), ...
            params.lpiter(sweep));
  end
  
  % Prune topics after a while
  if params.reduceTopics && sweep > params.reduceIter
    kinds = unique(params.Xpn.inds(:,3));
    params.Xpn.kinds = params.Xpn.kinds(kinds);
    params.Znk = params.Znk(:,kinds);
    params.Theta = params.Theta(kinds);
    params.Psi = params.Psi(kinds,:);
    params.Phi = params.Phi(:,kinds);
    params.psi_inds = params.psi_inds(kinds);
    if params.dosampleZnk
        params.Km = params.Km(kinds);
    end
    params.W = params.W(:,kinds);
    params.Zstar = params.Zstar(:,kinds);
    params.Lambda = params.Lambda(:,kinds);
  end

  % Collect samples if necessary
  if sweep > nburn && mod(sweep,thin) == 0
     %Theta_s{sidx} = params.Theta;
     %Phi_s{sidx} = params.Phi;
     %Psi_s{sidx} = params.Psi;
     %Znk_s{sidx} = params.Znk;
     %W_s{sidx} = params.W;
     %Lambda_s{sidx} = params.Lambda;
     %psi_inds_s{sidx} = params.psi_inds;
    
    if params.lpiter(sweep) > params.maxlp
        Theta_s = params.Theta;
        Phi_s = params.Phi;
        Psi_s = params.Psi;
        Znk_s = params.Znk;
        W_s = params.W;
        Lambda_s = params.Lambda;
        psi_inds_s = params.psi_inds;
        params.maxlp = params.lpiter(sweep);
    end

    % Compute running perplexity, based on Mingyuan's bnbp code...
    if params.computeperplex
        if params.soft
            for k = 1:K
                phiMat(:,k) = normcdf(params.Km{k}*params.W(:,k));
            end
            Mu = params.Phi*(diag(params.Theta)*phiMat'.*params.Psi);
        else
            Mu = params.Phi*(diag(params.Theta)*params.Znk'.*params.Psi);
        end
        % Compute sufficient statistic
        params.PoissMeanSS = params.PoissMeanSS + Mu;
        params.countSS = params.countSS + 1;

        % This is equivalent to what's in bnbp paper as the sum over k has
        % already happened
        temp = params.PoissMeanSS / params.countSS; % Division technically not needed
        temp = bsxfun(@rdivide, temp, sum(temp,1));

        params.perplex_vec(sidx) = ...
            sum(Xtest_sp(Yflagtest).*log(temp(Yflagtest))) / sum(Xtest_sp(:));
    end
    
    % Store log-probability of current sample
    params.lpsamp(sidx) = params.lpiter(sweep);
    sidx = sidx + 1;
  end
  
  if sweep > nburn && mod(sweep, params.save_freq) == 0
    save([params.smp_name '_' num2str(saveid) '.mat'], 'params');
    saveid = saveid + 1;
  end
  
end

% Save a final version with all samples
%save([save_prefix params.smp_name '_final.mat'], 'params');
if isfield(params, 'Km')
    params = rmfield(params, 'Km');
end

params.Theta_s = Theta_s;
params.Phi_s = Phi_s;
params.Psi_s = Psi_s;
params.Znk_s = Znk_s;
params.W_s = W_s;
params.Lambda_s = Lambda_s;
params.psi_inds_s = psi_inds_s;

params.perplex = params.perplex_vec(end);

gs = RandStream.getGlobalStream;
params.rng_seed = gs.Seed;
params.rng_type = gs.Type;
