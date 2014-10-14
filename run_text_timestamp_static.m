function run_text_timestamp_static(data_path, name)
% Time stamp prediction for text as static model.  We train a static
% gapp_pfa model at each observed timestamp and preduct the unobserved
% documents using that model.
%
% Inputs:
%   data_path : path to mat file with the data
%   name : name of the sampler for logs and temporary files

dbstop if error

addpath(genpath('~/matlab/lightspeed'));

s = RandStream('mt19937ar', 'Seed', 7654567);
RandStream.setGlobalStream(s);
gs = RandStream.getGlobalStream();
fprintf('Random seed: %d\n', gs.Seed);

%load data
counts = [];
words = [];
load(data_path);
words = vocab;
counts = termdoc;
clear termdoc vocab;

% Parse and standardize tstamps to [0,1]
tstamps = zeros(size(counts,2),1);
for i = 1:numel(time_labs)
  S = regexp(time_labs{i}, '_', 'split');
  tstamps(i) = str2num(S{4});
end
tstamps = tstamps - min(tstamps);
tstamps = tstamps / max(tstamps);

[counts,words,tstamps] = preprocess_corpus(counts, words, tstamps, 0.15, 20, 10); %#ok
[Ytrain,Ytest,Xtrain,Xtest] = holdout_docs(counts, 0.2, tstamps);

% Restructure data for sampler
[ii,jj,s] = find(Ytrain);
YYtrain = struct;
YYtrain.inds = [ii jj];
YYtrain.vals = s;
YYtrain.P = size(Ytrain,1);
YYtrain.N = size(Ytrain,2);
YYtrain.sum = sum(Ytrain(:));

[ii,jj,s] = find(Ytest);
YYtest = struct;
YYtest.inds = [ii jj];
YYtest.vals = s;
YYtest.P = size(Ytest,1);
YYtest.N = size(Ytest,2);
YYtest.sum = sum(Ytest(:));

tstamps_unq = unique(tstamps);

nburn_train = 500;
nsamp_train = 300;
nburn_test = 200;
nsamp_test = 100;
thin = 1;

P = size(Ytrain,1);  % Never changes.

%% Predict time stamps for held-out docs

% For each observed time stamp train a static model, and compute the
% log-liklihood of each test do with that model.  Afterwards, find the
% timestamp that maximized the likelihood for each document.
Ntrain = size(Ytrain,2);
Ntest = size(Ytest,2);
numtstamps = numel(tstamps_unq);
loglik = zeros(numtstamps, Ntest);
for i = 1:numtstamps
    
    fprintf('Training model on covariate %d of %d\n', i, numtstamps);
    
    N = Ntrain;
    K = 200;
    
    isstatic = true; % Whether we are initializing from a previous run

    sampleXpn = true;
    sampleTheta = true;
    samplePsi = true;
    samplePhi = true;
    sampleZnk = false;
    sampleW = false;
    sampleLambda = false;
    sampleKernWidth = false;

    Theta_a = 1/K;
    alpha_phi = .03; %0.01;
    c0 = 1e-4;
    d0 = 1e-4;
    % We want the entries of psi to be big
    e_psi = 10;%5;  % shape
    f_psi = .5;%.5; % rate

    mus = tstamps_unq;
    psi_dict = [5e-3 1e-3 5e-4 1e-4];
    psi_inds = ones(K,1);
    L = size(mus,1);
    Znk = ones(N, K);
    
    Theta_init = 1*ones(1,K); %.001*ones(1,K); %0.01*ones(1,K);
    % Draw columns of Phi_init from Dirichlet
    Phi_init = gamrnd(alpha_phi,1,P,K);
    % Set columns of Phi_init to uniform
    %Phi_init = ones(P,K);
    Phi_init = bsxfun(@rdivide, Phi_init, sum(Phi_init));

    Psi_init = .1*ones(K,N);
    W_init = zeros(L+1,K); %randn(L+1,K);
    %W_init(1,:) = -1;
    Lambda_init = (c0/d0)*ones(L+1,K);

    verbose = true;
    printfreq = 1;

    reduceTopics = true;
    reduceIter = 100;

    smp_name = name;
    save_freq = 0;
    perplex_soft = true;
    computeperplex = false;

    init = init_params_struct(YYtrain, Xtrain, 'nburn', nburn_train, ...
                              'nsamp', nsamp_train, ...
                              'thin', thin, ...
                              'dosampleXpn', sampleXpn, ...
                              'dosamplePsi', samplePsi, ...
                              'dosampleZnk', sampleZnk, ...
                              'dosampleTheta', sampleTheta, ...
                              'dosamplePhi', samplePhi, ...
                              'dosampleW', sampleW, ...
                              'dosampleLambda', sampleLambda, ...
                              'sampleKernWidth', sampleKernWidth, ...
                              'rng_seed', RandStream.getGlobalStream.Seed, ...
                              'Theta', Theta_init, ...
                              'Znk', Znk, ...
                              'Psi', Psi_init, ...
                              'Phi', Phi_init, ...
                              'W', W_init, ...
                              'Lambda', Lambda_init, ...
                              'mus', mus, ...
                              'psi_inds', psi_inds, ...
                              'psi_dict', psi_dict, ...
                              'Theta_a', Theta_a, ...
                              'alpha_phi', alpha_phi, ...
                              'e_psi', e_psi, 'f_psi', f_psi, ...
                              'c0', c0, 'd0', d0, ...
                              'isstatic', isstatic, ...
                              'verbose', verbose, 'printfreq', printfreq, ...
                              'reduceTopics', reduceTopics, ...
                              'reduceIter', reduceIter, ...
                              'computeperplex', computeperplex, ...
                              'soft', perplex_soft, ...
                              'smp_name', smp_name, 'save_freq', save_freq ...
                             );

    %% run model on train -- go read a paper...
    Xtest_str = struct; % Empty stuct is placeholder

    params = gapp_pfa_finite(YYtrain, Xtrain, Xtest_str, init);
    
    %%%------------------------------------------------------------------
    
    fprintf('Testing model on covariate %d of %d\n', i, numtstamps);
    
    % Update N to be number of test observations
    N = YYtest.N;
    K = size(params.Psi,1);

    isstatic = true; % Just always make this true

    sampleXpn = true;
    samplePsi = true;
    sampleTheta = false;
    samplePhi = false;
    sampleZnk = false;
    sampleW = false;
    sampleLambda = false;
    sampleKernWidth = false;

    Theta_a = 1/K;
    alpha_phi = 0.03;
    c0 = 1e-4;
    d0 = 1e-4;
    % We want the entries of psi to be big
    e_psi = 10;%5;  % shape
    f_psi = .5;%.5; % rate

    mus = tstamps_unq;
    psi_dict = [5e-3 1e-3 5e-4 1e-4];
    psi_inds = params.psi_inds_s;
    L = size(mus,1);

    % We use these below also
    Theta = params.Theta_s;
    Phi = params.Phi_s;
    W = params.W_s;
    Lambda = params.Lambda_s;
    Znk = ones(N, K);  % Static model so set to ones

    % Have to initialize this for the test data
    Psi_init = .1*ones(K,N);

    verbose = false;
    printfreq = 100;
    reduceTopics = false;
    reduceIter = 100;
    computeperplex = false;

    smp_name = name;
    save_freq = 0;

    timestamp = repmat(tstamps_unq(i,:), size(Xtest));
    
    init = init_params_struct(YYtest, timestamp, 'nburn', nburn_test, ...
                              'nsamp', nsamp_test, ...
                              'thin', thin, ...
                              'dosampleXpn', sampleXpn, ...
                              'dosampleZnk', sampleZnk, ...
                              'dosamplePsi', samplePsi, ...
                              'dosampleTheta', sampleTheta, ...
                              'dosamplePhi', samplePhi, ...
                              'dosampleW', sampleW, ...
                              'dosampleLambda', sampleLambda, ...
                              'sampleKernWidth', sampleKernWidth, ...
                              'rng_seed', RandStream.getGlobalStream.Seed, ...
                              'Theta', Theta, ...
                              'Psi', Psi_init, ...
                              'Znk', Znk, ...
                              'Phi', Phi, ...
                              'W', W, ...
                              'Lambda', Lambda, ...
                              'mus', mus, ...
                              'psi_inds', psi_inds, ...
                              'psi_dict', psi_dict, ...
                              'Theta_a', Theta_a, ...
                              'alpha_phi', alpha_phi, ...
                              'e_psi', e_psi, 'f_psi', f_psi, ...
                              'c0', c0, 'd0', d0, ...
                              'isstatic', isstatic, ...
                              'verbose', verbose, 'printfreq', printfreq, ...
                              'reduceTopics', reduceTopics, ...
                              'reduceIter', reduceIter, ...
                              'computeperplex', computeperplex, ...
                              'smp_name', smp_name, ...
                              'save_freq', save_freq ...
                             );

    % Empty Xtest as placeholder
    Xtest_str = struct;

    % Run sampler
    params_test = gapp_pfa_finite(YYtest, timestamp, Xtest_str, init);

    % Compute log-likelihood
    
    % Likilihood - may have to write a mex function for this
    %fprintf('likelihood...');
    
    Rate = Phi * (diag(Theta)*ones(size(params_test.Psi')).*params_test.Psi); % Parens matter here
    
    loglik(i,:) = sum(Ytest.*log(Rate + eps) - Rate - gammaln(Ytest + 1));
    
    clear Rate 
    
end

% Compute the index of the time stamp that gave the maximum for each data
% point
[~,maxind] = max(loglik);

% Predict time stamp
Xpred = tstamps_unq(maxind,:);

% L1 error, E[L1] and accuracy
L1 = sum(abs(Xtest - Xpred),2);
%L1err = sum(L1);
L1mean = mean(L1);
accuracy = sum(sum(abs(Xtest - Xpred),2) < 1e-4) / N;

% Evaluate baseline with Monte Carlo estimate over uniform assignments
MCSAMP = 500;
L1baseline = 0;
accuracybaseline = 0;
for i = 1:MCSAMP
    samps = randsample(numtstamps, N, true);
    L1base = sum(abs(Xtest - tstamps_unq(samps,:)),2);
    L1baseline = L1baseline + mean(L1base);
    accuracybaseline = accuracybaseline + sum(sum(abs(Xtest - tstamps_unq(samps,:)),2) < 1e-4) / N;
end

L1baseline = L1baseline / MCSAMP;
accuracybaseline = accuracybaseline / MCSAMP;

fprintf('Results:\n');
L1mean
L1baseline
accuracy
accuracybaseline
