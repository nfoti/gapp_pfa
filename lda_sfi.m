
clear;

addpath(genpath('~/work/text/tools/topictoolbox'));

%load data
counts = [];
words = [];
load('./SFI_data.mat');
tstamps = years;
clear years;

%recently been using q=0.3, .2 seems too harsh
[counts,words,tstamps] = preprocess_corpus(counts, words, tstamps, 0.3); 
[Ytrain,Ytest] = make_training(counts, 0.2);

Ytr_sp = sparse(Ytrain.inds(:,1), Ytrain.inds(:,2), Ytrain.vals, Ytrain.P, Ytrain.N);
Yte_sp = sparse(Ytest.inds(:,1), Ytest.inds(:,2), Ytest.vals, Ytest.P, Ytest.N);
[WS_tr,DS_tr] = SparseMatrixtoCounts(Ytr_sp);
[WS_te,DS_te] = SparseMatrixtoCounts(Yte_sp);

W = max(WS_tr);
D = max(DS_tr);

KK = [25 50 100 150 200 400 500]; %[100 200 300 400 500];
numK = numel(KK);

alpha = .1; %50/K; don't seem small enough, 
beta = .01; %200/W;
seed = 654345678;

nburn = 1000;
nsamp = 500;
thin = 1;

Zlast = cell(1, numK);
perplexes = zeros(1, numK);

parfor k = 1:numK
    
    K = KK(k);
    fprintf('K: %d\n', K);

    fprintf('Burn in...\n');
    [WP_tr,DP_tr,Z] = GibbsSamplerLDA(WS_tr, DS_tr, K, nburn, alpha, beta, seed, 0);

    WP_s = cell(1,nsamp);
    DP_s = cell(1,nsamp);
    Z_s = cell(1,nsamp);

    phi = zeros(W, K);
    theta = zeros(D, K);
    lpdocprob = zeros(1,D);
    Nm = sum(Yte_sp);

    fprintf('Sampling...\n');
    for i = 1:nsamp
      [WP,DP,Z] = GibbsSamplerLDA(WS_tr, DS_tr, K, thin, alpha, beta, seed, 0, Z);
      WP_s{i} = WP;
      DP_s{i} = DP;
      Z_s{i} = Z;
      
      phi = WP + beta;
      phi = bsxfun(@rdivide, phi, (sum(phi)+W*beta));

      theta = DP + alpha;
      theta = bsxfun(@rdivide, theta, (sum(theta,2)+K*alpha));
      
      %took out eps in log here since were no nonzeros anyways (checked)
      lpdocprob = lpdocprob + sum(Yte_sp.*log(phi*theta')); 

    end
    Zlast{k} = Z;

    % Compute perplexity
    lpdocprob = bsxfun(@rdivide, lpdocprob, nsamp);
    perplexes(k) = exp(-sum(lpdocprob)/sum(Nm));
end
=======
