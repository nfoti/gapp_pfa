
clear;

addpath(genpath('~/matlab/lightspeed2012b'));

% Set seed if desired
%s = RandStream('mt19937ar','Seed',8675309);
%s = RandStream('mt19937ar','Seed',5);
s = RandStream('mt19937ar','Seed',8365); % Good demo seed
%s = RandStream('mt19937ar','Seed',rand*10000);
RandStream.setGlobalStream(s);
gs = RandStream.getGlobalStream();
fprintf('Random seed: %d\n', gs.Seed);

plot_figs = 1;

Nx = 100;
d = 8; % edge length of images
K = d;
T = 20;
N = Nx*T;
Xunq = (1:T)';
X = zeros(N,1);
offs = 2;
for i = 1:size(Xunq,1)
  X((i-1)*Nx+(1:Nx)) = i + offs;
end

actual = struct; %store actual values of data to test perplexity

% Create "Topic" (bars that are normalized)
% Create bars
bars = zeros(d*d,d);
count = 1;
for i = 1:2:d
  b = zeros(d);
  b(i,:) = ones(1,d);
  bars(:,count) = reshape(b,d*d,1);
  count = count + 1;
end
for i = 2:2:d
  b = zeros(d);
  b(:,i) = ones(d,1);
  bars(:,count) = reshape(b,d*d,1);
  count = count + 1;
end
clear count b;
Phi = bars;       % Just so names match up with papers
clear bars;
Phi = bsxfun(@rdivide, Phi, sum(Phi,1));

actual.Phi = Phi;

Ktrue = size(Phi,2);
d = sqrt(size(Phi,1));

psi_w = 4;
Psi = psi_w*ones(Ktrue,N);
actual.Psi = Psi;

% Generate Nx data points per covariate location

% 1d covariates are the indices stored in the rows
mus = Xunq;
actual.mus = mus;
psi_dict = [3];
actual.psi_dict = psi_dict;
psi_inds = ones(Ktrue,1);
actual.psi_idx = psi_inds;
L = size(mus,1);
% Use spike-and-slab to draw weights here (hier. student-t used in sampler)
%Km = computeKernMats_exp2(X, K, mus, psi_inds, psi_dict);
Km = computeKernMats_exp2(X, struct('Psi',Psi, 'mus',mus, 'psi_inds',psi_inds, 'psi_dict',psi_dict));
W = zeros(L+1,Ktrue);
H = zeros(Ktrue,N);
sigma_weights = sqrt(4);
b_a = 1;
b_b = 1;
for k = 1:Ktrue
  nu_k = betarnd(b_a,b_b);
  inds = [true ; rand(L,1) < nu_k]; % bias weight always drawn from normal
  W(inds,k) = randn(sum(inds),1).*sigma_weights;
  W(~inds,k) = 0;
  tmp = Km{k}*W(:,k);
  tmin = min(tmp);
  if tmin < 0
    tmp = tmp+abs(tmin);
  end
  tmp = tmp./max(tmp);
  g_k = tmp;
  %g_k = normcdf(tmp./max(abs(tmp)));
  %g_k = normcdf(Km{k}*W(:,k));
  g_k(g_k==0) = 1e-16;
  g_k(g_k==1) = 1-1e-16;
  H(k,:) = rand(1,N) < g_k';
end
actual.W = W;

% !! Generate data
Y = poissrnd(Phi*(H.*Psi));  % This seems better
%Y = Phi*(H.*(Psi)*psi_w);

%Hold out some observations for time stamp prediction
testpct = .2;
nhold = floor(testpct * N);
rp = randperm(N);
test_inds = rp(1:nhold);
train_inds = setdiff(1:N, test_inds);

Ytrain = Y(:,train_inds);
Xtrain = X(train_inds,:);
Ytest = Y(:,test_inds);
Xtest = X(test_inds,:);


% Call YY b/c X are the covariates in this script
[ii,jj,s] = find(Ytrain);
YY = struct;
YY.inds = [ii jj];
YY.vals = s;
YY.P = size(Phi,1);
YY.N = size(Ytrain,2);
YY.sum = sum(Ytrain(:));
P = YY.P;
N = YY.N;

[ii,jj,s] = find(Ytest);
YYtest = struct;
YYtest.inds = [ii jj];
YYtest.vals = s;
YYtest.P = size(Phi,1);
YYtest.N = size(Ytest,2);
YYtest.sum = sum(Ytest(:));


% Initialize sampler

K = 20;

isstatic = true; % Just always make this true
nburn = 100;%1000;
nsamp = 100;%200;
thin = 1;

sampleXpn = true;
sampleTheta = true;
samplePhi = true;
sampleZnk = true;
sampleW = true;
sampleLambda = true;
sampleKernWidth = true;

Theta_a = 1/K;
% 1.01 works better, but 0.5 learns similar features
alpha_phi = 1.01;
c0 = 1e-4;
d0 = 1e-4;
% We want the entries of psi to be big
e_psi = 10;%5;  % shape
f_psi = .5;%.5; % rate

mus = Xunq;
psi_dict = [3 .5 .05];
psi_inds = ones(K,1);
L = size(mus,1);

Theta_init = .001*ones(1,K); %0.01*ones(1,K);
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
printfreq = 10;
reduceTopics = true;
reduceIter = 50;
computeperplex = false;

smp_name = 'bars_local';
save_freq = 0;


%Phi_init = Phi;  %%%%%%%%%%%%%%%%%%%%%

init = init_params_struct(YY, Xtrain, 'nburn', nburn, 'nsamp', nsamp, ...
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
                          'smp_name', smp_name, ...
                          'save_freq', save_freq ...
                         );
          
% Empty Xtest as placeholder
Xtest_str = struct;

% Run sampler
params = gapp_pfa_finite(YY, Xtrain, Xtest_str, init);

%% Predict time stamps for held-out documents
% Update parameters to only sample Znk and Psi for the test data and use
% the maximum probability samples from the training data

% Update N to be number of test observations
N = YYtest.N;
K = size(params.Psi,1);

isstatic = true; % Just always make this true
nburn = 100;%1000;
nsamp = 100;%200;
thin = 1;

sampleXpn = true;
sampleTheta = false;
samplePhi = false;
sampleZnk = true;
sampleW = false;
sampleLambda = false;
sampleKernWidth = false;

Theta_a = 1/K;
% 1.01 works better, but 0.5 learns similar features
alpha_phi = 1.01;
c0 = 1e-4;
d0 = 1e-4;
% We want the entries of psi to be big
e_psi = 10;%5;  % shape
f_psi = .5;%.5; % rate

mus = Xunq;
psi_dict = [3 .5 .05];
psi_inds = params.psi_inds_s;
L = size(mus,1);

% We use these below also
Theta = params.Theta_s;
Phi = params.Phi_s;
W = params.W_s;
Lambda = params.Lambda_s;

% Have to initialize this for the test data
Psi_init = .1*ones(K,N);

verbose = false;
printfreq = 100;
reduceTopics = false;
reduceIter = 0;
computeperplex = false;

smp_name = 'bars_local_timestamp';
save_freq = 0;

% For each observed time stamp assign the test data to that point and run
% the sampler.  Compute the log-likelihood for each test observation and
% for each observation store the time stamp that gave the maximum
% log-likelihood.
numXunq = numel(Xunq);
loglik = zeros(numXunq, N);
for i = 1:numXunq
    
    fprintf('Assigning test data to covariate %d of %d\n', i, numXunq);
    
    timestamp = repmat(Xunq(i,:), size(Xtest));
    
    init = init_params_struct(YYtest, timestamp, 'nburn', nburn, ...
                              'nsamp', nsamp, ...
                              'thin', thin, ...
                              'dosampleXpn', sampleXpn, ...
                              'dosampleZnk', sampleZnk, ...
                              'dosampleTheta', sampleTheta, ...
                              'dosamplePhi', samplePhi, ...
                              'dosampleW', sampleW, ...
                              'dosampleLambda', sampleLambda, ...
                              'sampleKernWidth', sampleKernWidth, ...
                              'rng_seed', RandStream.getGlobalStream.Seed, ...
                              'Theta', Theta, ...
                              'Psi', Psi_init, ...
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
    
    Km = computeKernMats_exp2(timestamp, params_test);
    
    Kplus = numel(Km);
    G = zeros(N,Kplus);
    for k = 1:Kplus
      G(:,k) = Km{k}*W(:,k);
    end

    % Likilihood - may have to write a mex function for this
    %fprintf('likelihood...');
    inds = YYtest.inds;
    vals = YYtest.vals;
    
    Rate = Phi * (diag(Theta)*normcdf(G').*params_test.Psi); % Parens matter here
    
    loglik(i,:) = sum(Ytest.*log(Rate + eps) - Rate - gammaln(Ytest + 1));
    
    clear Rate 
    
end

% Compute the index of the time stamp that gave the maximum for each data
% point
[~,maxind] = max(loglik);

% Predict time stamp
Xpred = Xunq(maxind,:);

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
    samps = randsample(numXunq, N, true);
    L1base = sum(abs(Xtest - Xunq(samps,:)),2);
    L1baseline = L1baseline + mean(L1base);
    accuracybaseline = accuracybaseline + sum(sum(abs(Xtest - Xunq(samps,:)),2) < 1e-4) / N;
end

L1baseline = L1baseline / MCSAMP;
accuracybaseline = accuracybaseline / MCSAMP;

fprintf('Results:\n');
L1mean
L1baseline
accuracy
accuracybaseline
