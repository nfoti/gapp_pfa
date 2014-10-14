function [err] = perplex_cluster(dataset, version, outpath, name)
% perplexity experiment for cluster
% nfolds to come later as input, for now all 1fold
%
% perplex_cluster(data_path, outpath, name, ntests)
% ntests: # of different values of alpha being tested currently; for now we
% do single folds but will modify soon to run multiple alphas for multiple
% folds
% 
% dataset: 1 for NIPS, 2 for SOTU, 3 for CONS
% version: 1 for dynamic, 2 for static, 3 for NMF
%

addpath(genpath('~/matlab/lightspeed2012b'));

s = RandStream('mt19937ar', 'Seed', 7654567);
RandStream.setGlobalStream(s);
gs = RandStream.getGlobalStream();
fprintf('Random seed: %d\n', gs.Seed);

%load data
fprintf('Processing corpus...\n');
counts = [];
tstamps = []; 
words = [];
% load(data_path);
% tstamps = years;
% clear years;

if dataset==1
    disp('NIPS');
    load('NIPS_data.mat');
    tstamps = years;
    clear years;
    
    %these settings yield ~1700 words, 1700 docs
    [counts,words,tstamps] = preprocess_corpus(counts, words, tstamps, ...
        0.15, 100, 10);
end

if dataset==2
    disp('SOTU');
    load('SOTU_data.mat');
    tstamps = years;
    clear years;
    
    %these settings yield ~1000 words, 6000 docs
    [counts,words,tstamps] = preprocess_corpus(counts, words, tstamps, ...
        0.15, 20, 10);
end

if dataset==3
    disp('CONS');
    load('CONS_data.mat');
    tstamps = years;
    clear years;
    
    %these settings yield ~1000 words, 3000 docs
    [counts,words,tstamps] = preprocess_corpus(counts, words, tstamps, ...
        0.3, 50, 10);
end

[Ytrain, Ytest] = make_training(counts, 0.2);

%will do folds later

% Ytrain = cell(1,nfolds);
% Ytest = cell(1,nfolds);
% for f = 1:nfolds
%     [Ytrain{f},Ytest{f}] = make_training(counts, 0.2);
% end

%% initialize parameters

P = Ytrain.P;
N = Ytrain.N;

isstatic = true; % Whether we are initializing from a previous run
nburn = 0;
nsamp = 1;
thin = 1; % Probably later

verbose = false;
printfreq = 1;

reduceTopics = true;
reduceIter = 1;

smp_name = name;
save_freq = 0;

logger = true;

computeperplex = true;
perplex_soft = false;

if version == 3
   K = [10 20 30 50 100 200 300]; %FIX QSUBARGS FOR THIS %CHECK
   ntests = numel(K);
else 
   K = 200; 
end

%set sampling logicals
if version == 1
   sampleXpn = true;
   sampleTheta = true;
   samplePhi = true;
   sampleZnk = true;
   sampleW = true;
   sampleLambda = true;
   sampleKernWidth = true;
elseif version == 2
   sampleXpn = true;
   sampleTheta = true;
   samplePhi = true;
   sampleZnk = false;
   sampleW = true;
   sampleLambda = true;
   sampleKernWidth = true;
else
   sampleXpn = true;
   sampleTheta = false;
   samplePhi = true;
   sampleZnk = false;
   sampleW = true;
   sampleLambda = true;
   sampleKernWidth = true;
end



% Set up cluster params
fprintf('Initializing cluster...\n');
try
  cluster = parcluster('anthill');
  anthill = 1;
catch ME  %#ok
  cluster = parcluster('local');
  anthill = 0;
end
pjob = createJob(cluster);

if anthill 
  % The -cwd is necessary to make sure that the workers start in the correct
  % directory
  qsubargs = '-cwd -l h_rt=23:59:59 -l virtual_free=2G';  %sets max runtime to 1 hr and memory used to 2G
  set(cluster,  'IndependentSubmitFcn', {@independentSubmitFcn, qsubargs})
end


%Set up runs depending on version we're running

%static or dynamic (alpha_phi will be vector, K constant)
if version == 1 || version == 2

    Theta_a = 1/K;
    alpha_phi = [.01 .03 .05 .1 .25 .5]; %LENGTH OF THIS VECTOR MUST MATCH NTESTS!
    ntests = numel(alpha_phi);
    
    c0 = 1e-4;
    d0 = 1e-4;
    % We want the entries of psi to be big
    e_psi = 10;%.9 NMF % shape
    f_psi = .5;%.1 NMF % rate

    mus = unique(tstamps);
    psi_dict = [1e-2 .5e-2 1e-3 .5e-3 1e-4];
    psi_inds = ones(K,1);
    L = size(mus,1);

    Theta_init = ones(1,K); %.001*ones(1,K); %0.01*ones(1,K);
    
    % Draw columns of Phi_init from Dirichlet
    for i=1:ntests
        Phi_init{i} = gamrnd(alpha_phi(i),1,P,K);
        Phi_init{i} = bsxfun(@rdivide, Phi_init{i}, sum(Phi_init{i}));
    end


    Psi_init = .1*ones(K,N);
    W_init = zeros(L+1,K); %randn(L+1,K);
    %W_init(1,:) = -1;
    Lambda_init = (c0/d0)*ones(L+1,K);
    
    % Initialize params struct and create jobs using the i'th training set
fprintf('Setting up jobs...\n');
for i = 1:ntests
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
                          'Phi', Phi_init{i}, ...
                          'W', W_init, ...
                          'Lambda', Lambda_init, ...
                          'mus', mus, ...
                          'psi_inds', psi_inds, ...
                          'psi_dict', psi_dict, ...
                          'Theta_a', Theta_a, ...
                          'alpha_phi', alpha_phi(i), ...
                          'e_psi', e_psi, 'f_psi', f_psi, ...
                          'c0', c0, 'd0', d0, ...
                          'isstatic', isstatic, ...
                          'verbose', verbose, 'printfreq', printfreq, ...
                          'reduceTopics', reduceTopics, ...
                          'reduceIter', reduceIter, ...
                          'smp_name', [smp_name '_j' num2str(i) '_'], ...
                          'soft', perplex_soft, 'logger', logger, ...
                          'computeperplex', computeperplex, ...
                          'save_freq', save_freq ...
                         );  
  createTask(pjob, @gapp_pfa_finite, 1, {Ytrain, tstamps, Ytest, init});
end
    
%NMF, alpha_phi constant, K vector
else
    alpha_phi = .05; 
    
    c0 = 1e-4;
    d0 = 1e-4;
    % We want the entries of psi to be big
    e_psi = .9;%.9 NMF % shape
    f_psi = .1;%.1 NMF % rate

    mus = unique(tstamps);
    psi_dict = [1e-2 .5e-2 1e-3 .5e-3 1e-4];
    L = size(mus,1);
    
    % Initialize params struct and create jobs using the i'th training set
    fprintf('Setting up jobs...\n');
    for i = 1:ntests
        Theta_a{i} = 1/K(i);
        psi_inds{i} = ones(K(i),1);

        Theta_init{i} = ones(1,K(i)); %.001*ones(1,K(i)); %0.01*ones(1,K(i));
         % Draw columns of Phi_init from Dirichlet

        Phi_init{i} = gamrnd(alpha_phi,1,P,K(i));
        % Set columns of Phi_init to uniform
        %Phi_init = ones(P,K(i));
        Phi_init{i} = bsxfun(@rdivide, Phi_init{i}, sum(Phi_init{i}));
        
        Psi_init{i} = .1*ones(K(i),N);
        W_init{i} = zeros(L+1,K(i)); %randn(L+1,K(i));
        %W_init(1,:) = -1;
        Lambda_init{i} = (c0/d0)*ones(L+1,K(i));
        
        
       
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
                              'Theta', Theta_init{i}, ...
                              'Psi', Psi_init{i}, ...
                              'Phi', Phi_init{i}, ...
                              'W', W_init{i}, ...
                              'Lambda', Lambda_init{i}, ...
                              'mus', mus, ...
                              'psi_inds', psi_inds{i}, ...
                              'psi_dict', psi_dict, ...
                              'Theta_a', Theta_a{i}, ...
                              'alpha_phi', alpha_phi, ...
                              'e_psi', e_psi, 'f_psi', f_psi, ...
                              'c0', c0, 'd0', d0, ...
                              'isstatic', isstatic, ...
                              'verbose', verbose, 'printfreq', printfreq, ...
                              'reduceTopics', reduceTopics, ...
                              'reduceIter', reduceIter, ...
                              'smp_name', [smp_name '_j' num2str(i) '_'], ...
                              'soft', perplex_soft, 'logger', logger, ...
                              'computeperplex', computeperplex, ...
                              'save_freq', save_freq ...
                             );  
        createTask(pjob, @gapp_pfa_finite, 1, {Ytrain, tstamps, Ytest, init});
    end
end


%% Run
fprintf('Running... come back tomorrow\n');
pjob.submit();
pjob.wait();

try
    result_params = pjob.fetchOutputs();  %#ok
catch ME
    tasks = get(pjob, 'Tasks');
    for t = 1:numel(tasks)
       err{t} = tasks(t).Error.stack
    end
%     delete(pjob); don't delete the job if error, want some task files
%     clear pjob;
    error('Errors occurred!');
end

% Don't do this in case it breaks.
% Tear down cluster NO!
% delete(pjob);
% clear pjob;

%% Save results
fprintf('Saving...\n');
whos result_params
save(outpath, 'result_params', 'words', 'tstamps', '-v7.3');

fprintf('Done!\n');

