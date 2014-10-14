function perplex_local(data_path, outpath, name)
% perplexity experiment
%
    
addpath(genpath('~/matlab/lightspeed'));

s = RandStream('mt19937ar', 'Seed', 7654567);
RandStream.setGlobalStream(s);
gs = RandStream.getGlobalStream();
fprintf('Random seed: %d\n', gs.Seed);

%load data
counts = [];
words = [];
load(data_path);
tstamps = years;
clear years;

[counts,words,tstamps] = preprocess_corpus(counts, words, tstamps, 0.15); %#ok
[Ytrain,Ytest] = make_training(counts, 0.2);

%% initialize parameters

K = 400;
P = Ytrain.P;
N = Ytrain.N;

isstatic = true; % Whether we are initializing from a previous run
nburn = 0; %2500
nsamp = 1; %2500
thin = 1; % Probably later

sampleXpn = true;
sampleTheta = false;%true;
samplePhi = true;
sampleZnk = true;
sampleW = true;
sampleLambda = true;
sampleKernWidth = true;

Theta_a = 1/K;
alpha_phi = 0.01;%.5; %0.01;
c0 = 1e-4;
d0 = 1e-4;
% We want the entries of psi to be big
e_psi = 10;%5;  % shape
f_psi = .5;%.5; % rate

mus = unique(tstamps);
psi_dict = [5 4 3 2 1];
psi_inds = ones(K,1);
L = size(mus,1);

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
computeperplex = true;
perplex_soft = false;

init = init_params_struct(Ytrain, tstamps, 'nburn', nburn, 'nsamp', nsamp, ...
                          'thin', thin, ...
                          'dosampleXpn', sampleXpn, ...
                          'dosampleZnk', sampleZnk, ...
                          'dosampleTheta', sampleTheta, ...
                          'dosamplePhi', samplePhi, ...
                          'dosampleW', sampleW, ...
                          'dosampleLambda', sampleLambda, ...
                          'sampleKernWidth', sampleKernWidth, ...
                          'rng_seed', RandStream.getGlobalStream.Seed, ...
                          'Theta', Theta_init, ...
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
result_params = {gapp_pfa_finite(Ytrain, tstamps, Ytest, init)};

fprintf('perplexity: %.2f\n', exp(-result_params{1}.perplex));

%SAVE to .mat file clearly labeled
save(outpath, 'result_params', 'words', 'tstamps');

