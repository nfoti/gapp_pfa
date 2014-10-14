% Run GaP-PFA sampler on sfi abstracts as a static topic model.


seed = 8675309;
s = RandStream('mt19937ar', 'Seed', seed);
RandStream.setGlobalStream(s);

prefix = '~/work/data/text/sfi_abstracts/WPData20120323';

% Filter out words and documents that won't be good.
% See the function for parameters.
% Variables that are loaded by the called script
X = [];
Y = [];
T = [];
vocab = [];
preprocess_sfi_corpus;

% X contains training data, Y contains testing

P = X.P;
N = X.N;

% Number of topics
K = 5; %400;

% Initialize parameters of sampler
isstatic = true;
nburn = 1000;
sampleXpn = true; % These settings define a static topic model
sampleTheta = true;
samplePhi = true;
sampleZnk = false;
sampleW = false;
sampleLambda = false;
sampleKernWidth = false;

Theta_a = 1/K;
alpha_phi = 1.01;
c0 = 1e-6;
d0 = 1e-6;

Znk = ones(N,K); % No thinning of topics
Theta = 0.01*ones(1,K); % Start all topic scores as small number
Phi = gamrnd(alpha_phi,1,P,K);
Phi = bsxfun(@rdivide, Phi, sum(Phi));
Psi = 0.01*ones(K,N);
% W has to have 1 more row than W
W = randn(11,K);  % Just make the following arbitrary b/c they're not used in static model
Lambda = ones(1,10);
mus = ones(10,1);
psi_inds = ones(1,K);
psi_dict = ones(1,10);
U = 1;

init = init_params_struct(X, T, 'nsamp', 1, 'thin', nburn, ...
                          'dosampleXpn', sampleXpn, ...
                          'dosampleZnk', sampleZnk, ...
                          'dosampleTheta', sampleTheta, ...
                          'dosamplePhi', samplePhi, ...
                          'dosampleW', sampleW, ...
                          'dosampleLambda', sampleLambda, ...
                          'sampleKernWidth', sampleKernWidth, ...
                          'rng_seed', RandStream.getGlobalStream.Seed, ...
                          'Znk', Znk, ...
                          'Theta', Theta, ...
                          'Psi', Psi, ...
                          'Phi', Phi, ...
                          'W', W, ...
                          'Lambda', Lambda, ...
                          'mus', mus, ...
                          'psi_inds', psi_inds, ...
                          'psi_dict', psi_dict, ...
                          'U', U, ...
                          'Theta_a', Theta_a, ...
                          'alpha_phi', alpha_phi, ...
                          'c0', c0, 'd0', d0, ...
                          'isstatic', isstatic, ...
                          'verbose', true ...
                         );
                            
% Run burn-in
fprintf('Burn-in\n');
%profile on;
params = gapp_pfa_finite(X, T, init);
%profile viewer;

% Set up sampler for collection phase

% Collect samples

% Compute perplexity

% Save results
